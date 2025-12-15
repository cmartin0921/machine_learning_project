import os
import csv
from pathlib import Path
from openaq import OpenAQ
from dotenv import load_dotenv

cur_dir = Path(__file__).resolve()
root_dir = cur_dir.parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path) 

# latitude, longitude
coordinates = (52.237049, 21.017532)
# coordinates = (136.90610, 35.14942)
# coordinates = (52.520008,13.404954) # Berlin
radius=10000
open_aq_limit=100
sensors_id=35986

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
client = OpenAQ(api_key=open_aq_api)
# TODO: eventually change this from asdf -> asdf.results
asdf = client.measurements.list(
    # coordinates=coordinates,
    # radius=radius,
    sensors_id=sensors_id,
    limit=open_aq_limit
)
# asdf = client.locations.list(
#     coordinates=coordinates,
#     radius=radius,
#     limit=open_aq_limit
# )

results_list = []
for r in asdf.results:
    row_results = {
        "datetime_from": r.period.datetime_from.utc,
        "datetime_to": r.period.datetime_to.utc,
        "value": r.value,
        "unit_name": r.parameter.name,
        "units": r.parameter.units,
    }
    results_list.append(row_results)

col_names = list(results_list[0].keys())
csv_file_name = 'dataset'
csv_file_dir = f'{root_dir}/data/raw/{csv_file_name}.csv'
file_exists = os.path.isfile(csv_file_dir)
with open(csv_file_dir, 'w', encoding="utf-8", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=col_names)
    if not file_exists:
        writer.writeheader()
    writer.writerows(results_list)

print("done")
client.close()


# Sensor IDS: 10776 / 35996,  10775 / 35986, 