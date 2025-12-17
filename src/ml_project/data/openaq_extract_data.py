import os
import csv
import time

from typing import Dict, Iterable, List, Tuple
from datetime import datetime, timezone

def openaq_extract_data(client, open_aq_cfg, directory_paths_dict):

    sensor_full_list = []
    # Step 1: Extract locations and sensors (within said locations) metadata.
    # The data for locations is written to a .csv file. Furthermore, a list
    # of sensor ids are stored separately.
    for location_data, sensor_list in _iter_locations(client, open_aq_cfg):
        if location_data is not None:
            locations_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["locations"]
            col_names = list(location_data.keys())
            file_exists = os.path.isfile(locations_file_loc)

            with open(locations_file_loc, "a", encoding="utf-8", newline="") as f:
                locations_writer = csv.DictWriter(f, fieldnames=col_names)

                if not file_exists:
                    locations_writer.writeheader()

                locations_writer.writerow(location_data)

        # Note: sensor_list is a List[Dict]; each sensor extract in the location from
        #       iter_locations is a Dict
        if len(sensor_list) > 0:
            sensors_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["sensors_metadata"]
            col_names = list(sensor_list[0].keys())
            file_exists = os.path.isfile(sensors_file_loc)

            with open(sensors_file_loc, "a", encoding="utf-8", newline="") as f:
                sensors_writer = csv.DictWriter(f, fieldnames=col_names)

                if not file_exists:
                    sensors_writer.writeheader()

                sensors_writer.writerows(sensor_list)

            # Do not need full information of sensors that was just written in .csv
            # Only need the sensor_id in order to extracts measurements from sensor via id
            sensor_id_list = [s["sensor_id"] for s in sensor_list]
            sensor_full_list.extend(sensor_id_list)

    if len(sensor_full_list) > 0:
        sensors_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["sensors_measurement"]
        file_exists = os.path.isfile(sensors_file_loc)

        with open(sensors_file_loc, "a", encoding="utf-8", newline="") as csvfile:
            for s_id in sensor_full_list:
                time.sleep(2)  # TODO: rate-limit handling
                to_write = _iter_sensor_measurements(client, open_aq_cfg, sensor_id=s_id)

                # Needed in order to get the keys that will be the header of the .csv file
                first_row = next(to_write, None)
                if first_row is not None:
                    col_names = list(first_row.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=col_names)

                    if not file_exists:
                        writer.writeheader()
                        file_exists = True

                    writer.writerow(first_row)
                    for row in to_write:
                        writer.writerow(row)





def _iter_locations(client, open_aq_cfg: Dict) -> Iterable[Tuple[dict, List[dict]]]:
    """Yield location row dicts that satisfy the configured date filter."""
    page = 1
    while True:
        location_response = client.locations.list(
            coordinates=(
                open_aq_cfg["coordinates"]["latitude"],
                open_aq_cfg["coordinates"]["longitude"],
            ),
            radius=open_aq_cfg["radius"],
            limit=open_aq_cfg["limit"],
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
            min_dt = open_aq_cfg["daterange"]["min"].replace(tzinfo=timezone.utc)
            if last_read >= min_dt:
                sensor_list = _extract_sensors_from_location(l)
                yield location_data_dict, sensor_list

        page += 1

def _iter_sensor_measurements(client, open_aq_cfg: Dict, sensor_id: int) -> Iterable[dict]:
    page = 1

    while True:
        time.sleep(0.5) # TODO: rate-limit handling
        sensor_data_response = client.measurements.list(
            sensors_id=sensor_id,
            datetime_from=open_aq_cfg["daterange"]["min"],
            datetime_to=open_aq_cfg["daterange"]["max"],
            limit=open_aq_cfg["limit"],
            rollup=open_aq_cfg["rollup"],
            page=page,
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
                "timestamp_rollup": open_aq_cfg["rollup"],
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