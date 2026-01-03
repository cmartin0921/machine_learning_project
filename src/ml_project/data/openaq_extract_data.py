import os
import csv
import time

from typing import Dict, Iterable, List, Tuple
from datetime import datetime, timezone

def openaq_extract_data(client, open_aq_cfg, directory_paths_dict):

    sensor_full_set = set()
    # Step 1: Extract locations and sensors (within said locations) metadata.
    # The data for locations is written to a .csv file. Furthermore, a list
    # of sensor ids are stored separately.
    locations_file_loc = directory_paths_dict["root"] / open_aq_cfg["outputs"]["dir"] / open_aq_cfg["outputs"]["files"]["openaq"]["locations"]
    sensors_meta_file_loc = directory_paths_dict["root"] / open_aq_cfg["outputs"]["dir"] / open_aq_cfg["outputs"]["files"]["openaq"]["sensors_metadata"]

    locations_file_exists = os.path.isfile(locations_file_loc)
    sensors_file_exists = os.path.isfile(sensors_meta_file_loc)

    with open(
        locations_file_loc, "a", encoding="utf-8", newline=""
    ) as location_f, open(
        sensors_meta_file_loc, "a", encoding="utf-8", newline=""
    ) as sensor_f:
        for location_data, sensor_list in _iter_locations(client, open_aq_cfg):
            if location_data is not None:
                location_col_names = list(location_data.keys())
                locations_writer = csv.DictWriter(location_f, fieldnames=location_col_names)
                if not locations_file_exists:
                    locations_writer.writeheader()
                    locations_file_exists = True

                locations_writer.writerow(location_data)

            if sensor_list is not None:
                sensor_col_names = list(sensor_list[0].keys())
                sensors_writer = csv.DictWriter(sensor_f, fieldnames=sensor_col_names)
                if not sensors_file_exists:
                    sensors_writer.writeheader()
                    sensors_file_exists = True

                sensors_writer.writerows(sensor_list)

                sensor_id_list = [s["sensor_id"] for s in sensor_list if s["sensor_id"] not in open_aq_cfg["sources"]["openaq"]["state"]["sensors_to_skip"]]
                sensor_full_set.update(sensor_id_list)

    # if len(sensor_full_set) > 0:
    #     sensors_file_loc = directory_paths_dict["root"] / open_aq_cfg["outputs"]["dir"] / open_aq_cfg["outputs"]["files"]["openaq"]["sensors_measurements"]
    #     file_exists = os.path.isfile(sensors_file_loc)

    #     with open(sensors_file_loc, "a", encoding="utf-8", newline="") as csvfile:
    #         for s_id in sorted(sensor_full_set):
    #             time.sleep(5)  # TODO: rate-limit handling
    #             if open_aq_cfg["sources"]["openaq"]["state"]["last_added_sensor_id"] is not None:
    #                 if open_aq_cfg["sources"]["openaq"]["state"]["last_added_sensor_id"] > s_id:
    #                     continue
                
    #             to_write = _iter_sensor_measurements(client, open_aq_cfg, sensor_id=s_id)

    #             # Needed in order to get the keys that will be the header of the .csv file
    #             first_row = next(to_write, None)
    #             if first_row is not None:
    #                 col_names = list(first_row.keys())
    #                 writer = csv.DictWriter(csvfile, fieldnames=col_names)

    #                 if not file_exists:
    #                     writer.writeheader()
    #                     file_exists = True

    #                 writer.writerow(first_row)
    #                 for row in to_write:
    #                     writer.writerow(row)

def _iter_locations(client, open_aq_cfg: Dict) -> Iterable[Tuple[dict, List[dict]]]:
    """Yield location row dicts that satisfy the configured date filter."""
    page = 1
    min_dt = open_aq_cfg["time"]["start"].replace(tzinfo=timezone.utc)
    for coord in open_aq_cfg["data"]["coordinates"]:
        while True:
            location_response = client.locations.list(
                coordinates=(
                    coord[0],
                    coord[1],
                ),
                radius=open_aq_cfg["data"]["radius"],
                limit=open_aq_cfg["paging"]["limit"],
                page=page,
            )

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
                    "country_id": l.country.id,
                    "country_name": l.country.name,
                    "country_code": l.country.code,
                    "timezone": l.timezone,
                    "first_read_at": l.datetime_first.utc,
                    "last_read_at": l.datetime_last.utc,
                }

                # Excludes sensors that do not have data within the date params passed
                last_read = datetime.fromisoformat(location_data_dict["last_read_at"])
                if last_read >= min_dt:
                    sensor_list = _extract_sensors_from_location(l)
                    yield location_data_dict, sensor_list

            page += 1


def _iter_sensor_measurements(client, open_aq_cfg: Dict, sensor_id: int) -> Iterable[dict]:
    page = 1
    while True:
        time.sleep(1.5) # TODO: rate-limit handling
        sensor_data_response = client.measurements.list(
            sensors_id=sensor_id,
            datetime_from=open_aq_cfg["time"]["start"],
            datetime_to=open_aq_cfg["time"]["end"],
            limit=open_aq_cfg["paging"]["limit"],
            rollup=open_aq_cfg["time"]["rollup"],
            page=page
        )
        
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

def _extract_sensors_from_location(location_result) -> List[dict]:
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