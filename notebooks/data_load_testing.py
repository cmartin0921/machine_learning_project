import os
import csv
import time
from dotenv import load_dotenv
from datetime import datetime, timezone
import yaml

from openaq import OpenAQ
from ml_project.utils import get_project_directories
from ml_project.data import (
    iter_locations,
    iter_sensor_measurements
)

directory_paths_dict = get_project_directories()
env_path = directory_paths_dict["root"] / ".env"
load_dotenv(dotenv_path=env_path)

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
client = OpenAQ(api_key=open_aq_api)

open_aq_yaml_file = directory_paths_dict["configs"] / "openaq.yaml"
with open_aq_yaml_file.open("r", encoding="utf-8") as f:
    open_aq_cfg = yaml.safe_load(f)["openaq"]

sensor_full_list = []
for location_row, sensor_list in iter_locations(client, open_aq_cfg):
    if location_row is not None:
        locations_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["locations"]
        col_names = list(location_row.keys())
        file_exists = os.path.isfile(locations_file_loc)

        with open(locations_file_loc, "a", encoding="utf-8", newline="") as f:
            locations_writer = csv.DictWriter(f, fieldnames=col_names)

            if not file_exists:
                locations_writer.writeheader()

            locations_writer.writerow(location_row)

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

for s_id in sensor_full_list:
    time.sleep(2)
    sensor_measurement_row = iter_sensor_measurements(client, open_aq_cfg, sensor_id=s_id)

    to_add = next(sensor_measurement_row, None)
    if to_add is not None:
        sensors_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["sensors_measurement"]
        col_names = to_add.keys()
        file_exists = os.path.isfile(sensors_file_loc)

        with open(sensors_file_loc, 'a', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_names)

            # Only writes the header on first creation
            if not file_exists:
                writer.writeheader()

            for r in to_add:
                writer.writerow(to_add)

# sensor_data_list = []
# for s in sensor_ids:
#     time.sleep(3)
#     sensors_page_counter = 1    # Bookkeeping for the number of pages parsed.

#     while True:
#         sensor_response = client.measurements.list(
#             sensors_id=s,
#             datetime_from=open_aq_cfg["daterange"]["min"],
#             datetime_to=open_aq_cfg["daterange"]["max"],
#             limit=open_aq_cfg["limit"],
#             rollup=open_aq_cfg["rollup"],
#             page=sensors_page_counter
#         )
#         print(f"Sensor ID: {s} at page {sensors_page_counter} with results length of {len(sensor_response.results)}")
#         # Loop exits when results from sensor_response is empty
#         if not sensor_response.results:
#             break

#         for sd in sensor_response.results:
#             sensor_data_row_dict = {
#                 "sensor_id": s,
#                 "datetime_from": sd.period.datetime_from.utc,
#                 "datetime_to": sd.period.datetime_to.utc,
#                 "timestamp_rollup": open_aq_cfg["rollup"],
#                 "value": sd.value,
#                 "metric_name": sd.parameter.name,
#                 "units": sd.parameter.units,
#             }
#             sensor_data_list.append(sensor_data_row_dict)

#         sensors_page_counter += 1

#     # Writing in bulk after every sensor measurement (if exists)
#     if len(sensor_data_list) > 0:
#         sensors_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["sensors"]
#         col_names = list(sensor_data_list[0].keys())
#         file_exists = os.path.isfile(sensors_file_loc)
#         with open(sensors_file_loc, 'a', encoding="utf-8", newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=col_names)

#             # Only writes the header on first creation
#             if not file_exists:
#                 writer.writeheader()
#             writer.writerows(sensor_data_list)

print("done")
client.close()

#TODO: handle rate limits and timeouts