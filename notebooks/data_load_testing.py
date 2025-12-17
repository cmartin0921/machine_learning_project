import os
import csv
import time
from dotenv import load_dotenv
from datetime import datetime, timezone
import yaml

from openaq import OpenAQ
from ml_project.utils import (
    get_project_directories,
    write_to_csv
)
from ml_project.data import (
    openaq_extract_data
)

directory_paths_dict = get_project_directories()
env_path = directory_paths_dict["root"] / ".env"
load_dotenv(dotenv_path=env_path)

open_aq_api = os.getenv("OPEN_AQ_API_KEY")
open_aq_client = OpenAQ(api_key=open_aq_api)

open_aq_yaml_file = directory_paths_dict["configs"] / "openaq.yaml"
with open_aq_yaml_file.open("r", encoding="utf-8") as f:
    open_aq_cfg = yaml.safe_load(f)["openaq"]

openaq_extract_data(open_aq_client, open_aq_cfg, directory_paths_dict)


print("done")
open_aq_client.close()

#TODO: handle rate limits and timeouts