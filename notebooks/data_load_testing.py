import os
import csv
from openaq import OpenAQ
from dotenv import load_dotenv
from ml_project.utils import get_project_directories

directory_paths_dict = get_project_directories()
env_path = directory_paths_dict["root"] / ".env"
load_dotenv(dotenv_path=env_path)

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
client = OpenAQ(api_key=open_aq_api)

# latitude, longitude
coordinates = (52.237049, 21.017532)
radius=10000
open_aq_limit=3
# sensors_id=35986

locations_result = client.locations.list(
    coordinates=coordinates,
    radius=radius,
    limit=open_aq_limit
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

col_names = list(location_data_list[0].keys())
csv_file_name = 'location_dataset'
csv_file_dir = f'{root_dir}/data/raw/{csv_file_name}.csv'
file_exists = os.path.isfile(csv_file_dir)
with open(csv_file_dir, 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=col_names)
    if not file_exists:
        writer.writeheader()
    writer.writerows(location_data_list)


sensor_data_list = []
for s in sensor_ids:
    sensor_result = client.measurements.list(
        sensors_id=s,
        limit=open_aq_limit
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

col_names = list(sensor_data_list[0].keys())
csv_file_name = 'sensor_dataset'
csv_file_dir = f'{root_dir}/data/raw/{csv_file_name}.csv'
file_exists = os.path.isfile(csv_file_dir)
with open(csv_file_dir, 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=col_names)
    if not file_exists:
        writer.writeheader()
    writer.writerows(sensor_data_list)

print("done")
client.close()


# Sensor IDS: 10776 / 35996,  10775 / 35986, 