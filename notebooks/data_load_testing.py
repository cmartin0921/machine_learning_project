import os
import csv
from openaq import OpenAQ
from dotenv import load_dotenv
import yaml
from ml_project.utils import get_project_directories

directory_paths_dict = get_project_directories()
env_path = directory_paths_dict["root"] / ".env"
load_dotenv(dotenv_path=env_path)

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
client = OpenAQ(api_key=open_aq_api)


open_aq_yaml_file = directory_paths_dict["configs"] / "openaq.yaml"
with open_aq_yaml_file.open("r", encoding="utf-8") as f:
    open_aq_cfg = yaml.safe_load(f)["openaq"]

locations_result = client.locations.list(
    coordinates=(
        open_aq_cfg["coordinates"]["latitude"],
        open_aq_cfg["coordinates"]["longitude"]
    ),
    radius=open_aq_cfg["radius"],
    limit=open_aq_cfg["limit"]
).results

location_data_list = []
sensor_ids = []
for l in locations_result:
    sensor_data_list = []
    for s in l.sensors:
        sensor_row_dict = {
            "sensor_id": s.id,
            "measurement": s.parameter.display_name,
            "measurement_name": s.parameter.name,
            "units": s.parameter.units,
        }

        sensor_ids.append(s.id)
        sensor_data_list.append(sensor_row_dict)

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
    location_data_list.append(location_row_dict)

locations_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["locations"]
with open(locations_file_loc, 'w', encoding="utf-8", newline='') as csvfile:
    col_names = list(location_data_list[0].keys())

    writer = csv.DictWriter(csvfile, fieldnames=col_names)
    writer.writeheader()
    writer.writerows(location_data_list)


sensor_data_list = []
for s in sensor_ids:
    sensor_result = client.measurements.list(
        sensors_id=s,
        limit=open_aq_cfg["limit"]
    )
    for sd in sensor_result.results:
        sensor_data_row_dict = {
            "sensor_id": s,
            "datetime_from": sd.period.datetime_from.utc,
            "datetime_to": sd.period.datetime_to.utc,
            "value": sd.value,
            "unit_name": sd.parameter.name,
            "units": sd.parameter.units,
        }
        sensor_data_list.append(sensor_data_row_dict)

sensors_file_loc = directory_paths_dict["data_raw"] / open_aq_cfg["outputs"]["sensors"]
with open(sensors_file_loc, 'w', encoding="utf-8", newline='') as csvfile:
    col_names = list(sensor_data_list[0].keys())

    writer = csv.DictWriter(csvfile, fieldnames=col_names)
    writer.writeheader()
    writer.writerows(sensor_data_list)

print("done")
client.close()


# Sensor IDS: 10776 / 35996,  10775 / 35986, 