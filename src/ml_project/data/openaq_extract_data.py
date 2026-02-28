import os
import csv
import time
import re
import logging

from typing import Dict, Iterable, List, Tuple
from datetime import datetime, timezone

from openaq.shared.exceptions import ServerError, RateLimitError
from httpx import ReadTimeout
from ml_project.utils import write_csv

logger = logging.getLogger("ml_project")

def openaq_extract_data(
    client,
    open_aq_cfg: Dict,
    directory_paths_dict
):
        """Extract locations and sensor measurements and write to CSV.

        Parameters
        - client: OpenAQ client instance providing `locations` and
            `measurements` endpoints.
        - open_aq_cfg (dict): Configuration dictionary containing
            `data`, `time`, `paging`, and `outputs` entries.
        - directory_paths_dict (Path-like): Paths used for output files.

        The function writes three CSV files (locations, sensors metadata,
        and sensors measurements) under the configured output directory.
        It yields no value; side effects are writing files to disk.
        """

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
                    logger.debug("Successfully wrote data to %s.", locations_file_loc)
                else:
                    logger.debug("No data passed to write to %s.", locations_file_loc)

                if write_csv(sensor_f, sensor_list):
                    logger.debug("Successfully wrote data to %s.", sensors_meta_file_loc)
                else:
                    logger.debug("No data passed to write to %s.", sensors_meta_file_loc)

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
                    logger.debug("Successfully wrote %d rows to %s for sensor ID %d.", rows_written, sensors_file_loc, s_id)
                else:
                    logger.warning("No data passed to write to %s for sensor ID %d.", sensors_file_loc, s_id)

def _iter_locations(
    client,
    open_aq_cfg: Dict
) -> Iterable[Tuple[dict, List[dict]]]:
    """Yield tuples of (location_data, sensor_list) for coordinates.

    Each yielded item contains a dictionary of location metadata and a
    list of sensor metadata dictionaries for that location. Locations
    are filtered so that only those with recent data (after
    `open_aq_cfg['time']['start']`) are yielded.
    """

    min_dt = open_aq_cfg["time"]["start"].replace(tzinfo=timezone.utc)

    for coord in open_aq_cfg["data"]["coordinates"]:
        page = 1

        while True:
            try:
                logger.info("Extracting data for location with coordinates [%f, %f]", coord[0], coord[1])
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
                logger.error("Failed to extract for the following coordinates [%f, %f] on page %d. Full error: %s", coord[0], coord[1], page, server_err)
                break
            except RateLimitError as rate_err:
                rate_sleep = 70
                match = re.search(r"(\d+)\s*seconds?", str(rate_err))
                if match:
                    rate_sleep = int(match.group(1))
                logger.error("Rate Limit Error: %s. Resuming data extraction from API in %d seconds.", rate_err, rate_sleep + 5)
                time.sleep(rate_sleep + 5)
                continue
            except ReadTimeout as timeout_err:
                logger.error("Read Timeout Error: %s. Resuming data extraction from API in %d seconds.", timeout_err, 10)
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
                        logger.info("Extracted %d sensor metadata data for location ID %d from coordinates [%f, %f]", len(sensor_list), l.id, coord[0], coord[1])
                        yield location_data_dict, sensor_list
                    else:
                        logger.debug("Skipping location ID %d - last read %s is before %s", l.id, last_read, min_dt)
                else:
                    logger.debug("Skipping location ID %d - no last_read_at timestamp", l.id)
            
            # Prevents from potentially pagination to a page with 0 results
            # if it is known the current page did not hit the limit
            if len(location_response.results) == open_aq_cfg["paging"]["limit"]:
                page += 1
            else:
                break


def _iter_sensor_measurements(
        client,
        open_aq_cfg: Dict,
        sensor_id: int
) -> Iterable[dict]:
    """Yield measurement dicts for a given `sensor_id`.

    Each yielded dictionary contains measurement metadata such as the
    rollup timestamps, value, metric name, and units. Pagination and
    basic retry handling for common OpenAQ client errors are performed
    internally.
    """
    page = 1
    while True:
        logger.info("Fetching data for sensor ID %d between %s and %s at %s granularity on page %d", sensor_id, open_aq_cfg["time"]["start"], open_aq_cfg["time"]["end"], open_aq_cfg["time"]["rollup"], page)
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
        except ServerError as server_err:
            logger.error("Failed to extract for the following sensor ID %d on page %d. Full error: %s", sensor_id, page, server_err)
            break
        except RateLimitError as rate_err:
            rate_sleep = 70
            match = re.search(r"(\d+)\s*seconds?", str(rate_err))
            if match:
                rate_sleep = int(match.group(1))
            logger.error("Rate Limit Error: %s. Resuming data extraction from API in %d seconds.", rate_err, rate_sleep + 5)
            time.sleep(rate_sleep + 5)
            continue
        except ReadTimeout as timeout_err:
            logger.error("Read Timeout Error: %s. Resuming data extraction from API in %d seconds.", timeout_err, 10)
            time.sleep(10)
            continue
        except Exception as err:
            logger.error("Unknown Error: %s. Resuming data extraction from API in %d seconds.", err, 10)
            time.sleep(10)
            continue
        
        logger.info("Sensor ID: %d at page %d with results length of %d", sensor_id, page, len(sensor_data_response.results))

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

        # Prevents from potentially pagination to a page with 0 results
        # if it is known the current page did not hit the limit
        if len(sensor_data_response.results) == open_aq_cfg["paging"]["limit"]:
            page += 1
        else:
            break

def _extract_sensors_from_location(
    location_result
) -> List[dict]:
    """Return a list of sensor metadata dicts for a location result.

    The returned list contains dictionaries with fields `sensor_id`,
    `measurement`, `measurement_name`, `units`, and `location_id`.
    """

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