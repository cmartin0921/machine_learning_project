import os
import csv
import time
import re

from typing import Dict, Iterable, List, Tuple
from datetime import datetime, timezone

from openaq.shared.exceptions import ServerError, RateLimitError
from httpx import ReadTimeout
from ml_project.utils import write_csv

def openaq_extract_data(
    client,
    open_aq_cfg,
    directory_paths_dict
):

    sensor_full_set = set()
    # Step 1: Extract locations and sensors (within said locations) metadata.
    # The data for locations is written to a .csv file. Furthermore, a list
    # of sensor ids are stored separately.
    locations_file_loc = directory_paths_dict["root"] / open_aq_cfg["outputs"]["dir"] / open_aq_cfg["outputs"]["files"]["openaq"]["locations"]
    sensors_meta_file_loc = directory_paths_dict["root"] / open_aq_cfg["outputs"]["dir"] / open_aq_cfg["outputs"]["files"]["openaq"]["sensors_metadata"]

    locations_file_exists = os.path.isfile(locations_file_loc)
    sensors_file_exists = os.path.isfile(sensors_meta_file_loc)

    # Only go through OpenAQ if files do not exist internally
    if not (locations_file_exists and sensors_file_exists):
        with open(
            locations_file_loc, "a", encoding="utf-8", newline=""
        ) as location_f, open(
            sensors_meta_file_loc, "a", encoding="utf-8", newline=""
        ) as sensor_f:
            for location_data, sensor_list in _iter_locations(client, open_aq_cfg):
                if write_csv(location_f, location_data):
                    print(f"Successfully wrote data to {locations_file_loc}.")
                else:
                    print(f"No data passed to write to {locations_file_loc}.")

                if write_csv(sensor_f, sensor_list):
                    print(f"Successfully wrote data to {sensors_meta_file_loc}.")
                else:
                    print(f"No data passed to write to {sensors_meta_file_loc}.")

                sensor_id_list = [s["sensor_id"] for s in sensor_list if s["sensor_id"] not in open_aq_cfg["sources"]["openaq"]["state"]["sensors_to_skip"]]
                sensor_full_set.update(sensor_id_list)
    else:
        # Read sensors from existing data
        with open(
            sensors_meta_file_loc, "r", encoding="utf-8", newline=""
        ) as sensor_f:
            sensor_csv_data = csv.DictReader(sensor_f)
            for r in sensor_csv_data:
                sensor_full_set.add(int(r["sensor_id"]))

    if len(sensor_full_set) > 0:
        sensors_file_loc = directory_paths_dict["root"] / open_aq_cfg["outputs"]["dir"] / open_aq_cfg["outputs"]["files"]["openaq"]["sensors_measurements"]

        with open(sensors_file_loc, "a", encoding="utf-8", newline="") as sensor_measurement_f:
            for s_id in sorted(sensor_full_set):
                if open_aq_cfg["sources"]["openaq"]["state"]["last_added_sensor_id"] is not None:
                    if open_aq_cfg["sources"]["openaq"]["state"]["last_added_sensor_id"] > s_id:
                        continue

                rows_written = 0
                for measurement in _iter_sensor_measurements(client, open_aq_cfg, sensor_id=s_id):
                    if write_csv(sensor_measurement_f, measurement):
                        rows_written += 1
                
                if rows_written > 0:
                    print(f"Successfully wrote {rows_written} rows to {sensors_file_loc} for {s_id}.")
                else:
                    print(f"No data passed to write to {sensors_file_loc} for {s_id}.")

def _iter_locations(client, open_aq_cfg: Dict) -> Iterable[Tuple[dict, List[dict]]]:
    page = 1
    min_dt = open_aq_cfg["time"]["start"].replace(tzinfo=timezone.utc)

    for coord in open_aq_cfg["data"]["coordinates"]:
        page = 1

        while True:
            try:
                location_response = client.locations.list(
                    coordinates=(
                        coord[0],
                        coord[1],
                    ),
                    radius=open_aq_cfg["data"]["radius"],
                    limit=open_aq_cfg["paging"]["limit"],
                    page=page,
                )
            except ServerError as server_err:
                print(f"Failed to extract for the following: coord {coord} on page {page}. Full error: {server_err}")
                break
            except RateLimitError as rate_err:
                print(f"Rate Limit Error: {rate_err}")
                rate_sleep = 70
                match = re.search(r"(\d+)\s*seconds?", str(rate_err))
                if match:
                    rate_sleep = int(match.group(1))
                time.sleep(rate_sleep + 5)
                continue
            except ReadTimeout as timeout_err:
                print(f"Read Timeout Error: {timeout_err}")
                time.sleep(10)
                continue
            
            # Exists when there are no longer any results from pagination
            if not location_response.results:
                break

            for l in location_response.results:
                location_data_dict = {
                    "location_id": l.id,
                    "location_name": l.name,
                    "location_owner_name": l.owner.name,
                    "location_owner_id": l.owner.id,
                    "latitude": l.coordinates.latitude,
                    "longitude": l.coordinates.longitude,
                    "city_name": l.locality,
                    "city_latitude": coord[0],
                    "city_longitude": coord[1],
                    "country_id": l.country.id,
                    "country_name": l.country.name,
                    "country_code": l.country.code,
                    "timezone": l.timezone,
                    "first_read_at": None if l.datetime_first is None else l.datetime_first.utc,
                    "last_read_at": None if l.datetime_last is None else l.datetime_last.utc,
                }

                # Excludes sensors that do not have data within the date params passed
                if location_data_dict["last_read_at"] is not None:
                    last_read = datetime.fromisoformat(location_data_dict["last_read_at"])

                    if last_read >= min_dt:
                        sensor_list = _extract_sensors_from_location(l)
                        yield location_data_dict, sensor_list
            
            page += 1


def _iter_sensor_measurements(
        client,
        open_aq_cfg: Dict,
        sensor_id: int
) -> Iterable[dict]:
    
    page = 1
    while True:
        print(f"Fetching data for sensor ID {sensor_id} between {open_aq_cfg["time"]["start"]} and {open_aq_cfg["time"]["end"]} at {open_aq_cfg["time"]["rollup"]} granularity on page {page}")
        time.sleep(0.5)
        try:
            sensor_data_response = client.measurements.list(
                sensors_id=sensor_id,
                datetime_from=open_aq_cfg["time"]["start"],
                datetime_to=open_aq_cfg["time"]["end"],
                limit=open_aq_cfg["paging"]["limit"],
                rollup=open_aq_cfg["time"]["rollup"],
                page=page
            )
        except ServerError as int_err:
            print(f"Server Error: {int_err}")
            break
        except RateLimitError as rate_error:
            print(f"Rate Limit Error: {rate_error}")
            rate_sleep = 70
            match = re.search(r"(\d+)\s*seconds?", str(rate_error))
            if match:
                rate_sleep = int(match.group(1))
            time.sleep(rate_sleep + 5)
            continue
        except ReadTimeout as timeout_err:
            print(f"Read Timeout Error: {timeout_err}")
            time.sleep(10)
            continue
        except Exception as err:
            print(f"Unknown Error: {err}")
            time.sleep(10)
            continue
        
        print(f"Sensor ID: {sensor_id} at page {page} with results length of {len(sensor_data_response.results)}")

        # Exists when there are no longer any results from pagination
        if not sensor_data_response.results:
            break

        for sd in sensor_data_response.results:
            sensor_measurement = {
                "sensor_id": sensor_id,
                "datetime_from": sd.period.datetime_from.utc,
                "datetime_to": sd.period.datetime_to.utc,
                "timestamp_rollup": open_aq_cfg["time"]["rollup"],
                "value": sd.value,
                "metric_name": sd.parameter.name,
                "units": sd.parameter.units,
            }

            yield sensor_measurement

        page += 1

def _extract_sensors_from_location(
        location_result
) -> List[dict]:
    
    """Extract sensors metadata rows from a single location result object."""
    sensor_location_list = []
    for s in location_result.sensors:
        sensor_location_list.append({
            "sensor_id": s.id,
            "measurement": s.parameter.display_name,
            "measurement_name": s.parameter.name,
            "units": s.parameter.units,
            "location_id": location_result.id,
        })

    return sensor_location_list