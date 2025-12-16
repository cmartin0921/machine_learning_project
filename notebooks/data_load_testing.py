import os
import csv
import time
from dotenv import load_dotenv
from datetime import datetime, timezone
import yaml

from openaq import OpenAQ
from ml_project.utils import get_project_directories

directory_paths_dict = get_project_directories()
env_path = directory_paths_dict["root"] / ".env"
load_dotenv(dotenv_path=env_path)

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
client = OpenAQ(api_key=open_aq_api)

open_aq_yaml_file = directory_paths_dict["configs"] / "openaq.yaml"
with open_aq_yaml_file.open("r", encoding="utf-8") as f:
    open_aq_cfg = yaml.safe_load(f)["openaq"]

# Assumption: only one location at a time
locations_page_counter = 1
while True:
    locations_response = client.locations.list(
        coordinates=(
            open_aq_cfg["coordinates"]["latitude"],
            open_aq_cfg["coordinates"]["longitude"]
        ),
        radius=open_aq_cfg["radius"],
        limit=open_aq_cfg["limit"],
        page=locations_page_counter
    )

    # Loop exits when results from locations_response is empty
    if not locations_response.results:
        break

    location_data_list = []
    sensor_ids = []
    sensor_data_list = []
    for l in locations_response.results:
        location_row_dict = {
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

        # Only take in sensor location data if it contains data in the specified date range in params
        if datetime.fromisoformat(location_row_dict["last_read_at"].replace("Z", "+00:00")) >= open_aq_cfg["daterange"]["min"].replace(tzinfo=timezone.utc):
            for s in l.sensors:
                sensor_row_dict = {
                    "sensor_id": s.id,
                    "measurement": s.parameter.display_name,
                    "measurement_name": s.parameter.name,
                    "units": s.parameter.units,
                    "location_id": l.id
                }

                sensor_ids.append(s.id)
                sensor_data_list.append(sensor_row_dict)

            location_data_list.append(location_row_dict)

    locations_page_counter += 1

    if len(location_data_list) > 0:
        locations_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["locations"]
        with open(locations_file_loc, 'a', encoding="utf-8", newline='') as csvfile:
            col_names = list(location_data_list[0].keys())

            writer = csv.DictWriter(csvfile, fieldnames=col_names)
            writer.writeheader()
            writer.writerows(location_data_list)


sensor_data_list = []
for s in sensor_ids:
    time.sleep(3)
    sensors_page_counter = 1    # Bookkeeping for the number of pages parsed.

    while True:
        sensor_response = client.measurements.list(
            sensors_id=s,
            datetime_from=open_aq_cfg["daterange"]["min"],
            datetime_to=open_aq_cfg["daterange"]["max"],
            limit=open_aq_cfg["limit"],
            rollup=open_aq_cfg["rollup"],
            page=sensors_page_counter
        )
        print(f"Sensor ID: {s} at page {sensors_page_counter} with results length of {len(sensor_response.results)}")
        # Loop exits when results from sensor_response is empty
        if not sensor_response.results:
            break

        for sd in sensor_response.results:
            sensor_data_row_dict = {
                "sensor_id": s,
                "datetime_from": sd.period.datetime_from.utc,
                "datetime_to": sd.period.datetime_to.utc,
                "timestamp_rollup": open_aq_cfg["rollup"],
                "value": sd.value,
                "metric_name": sd.parameter.name,
                "units": sd.parameter.units,
            }
            sensor_data_list.append(sensor_data_row_dict)

        sensors_page_counter += 1

    # Writing in bulk after every sensor measurement (if exists)
    if len(sensor_data_list) > 0:
        sensors_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["sensors"]
        col_names = list(sensor_data_list[0].keys())
        file_exists = os.path.isfile(sensors_file_loc)
        with open(sensors_file_loc, 'a', encoding="utf-8", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_names)

            # Only writes the header on first creation
            if not file_exists:
                writer.writeheader()
            writer.writerows(sensor_data_list)

print("done")
client.close()

#TODO: handle rate limits and timeouts