import os
from ml_project.utils import write_csv

def meteostat_extract_data(client, cfg, directory_paths_dict):
    
    outputs_dir = cfg["outputs"]["dir"]
    weather_filename = cfg["outputs"]["files"]["meteostat"]["weather_daily"]
    weather_file_loc = directory_paths_dict["root"] / outputs_dir / weather_filename

    file_exists = os.path.isfile(weather_file_loc)

    if file_exists:
        print(f"Weather file already exists at {weather_file_loc}. Skipping extraction.")
        return

    with open(weather_file_loc, "a", encoding="utf-8", newline="") as weather_f:
        rows_written = 0
        for record in client.extract_daily_data(cfg):
            if write_csv(weather_f, record):
                rows_written += 1
        
        if rows_written > 0:
            print(f"Successfully wrote {rows_written} rows to {weather_file_loc}.")
        else:
            print(f"No weather data written to {weather_file_loc}.")