import os
from pathlib import Path
from openaq import OpenAQ
from dotenv import load_dotenv

cur_dir = Path(__file__).resolve().parent
env_path = cur_dir.parent / ".env"
load_dotenv(dotenv_path=env_path) 

# latitude, longitude
coordinates = (52.237049, 21.017532)
# coordinates = (136.90610, 35.14942)
# coordinates = (52.520008,13.404954) # Berlin
radius=10000
open_aq_limit=10
sensors_id=35996

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
client = OpenAQ(api_key=open_aq_api)
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
print(asdf.results)
print("done")
client.close()


# Sensor IDS: 10776 / 35996,  10775 / 35986, 