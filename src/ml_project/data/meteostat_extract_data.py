import os
import logging
from ml_project.utils import write_csv

logger = logging.getLogger("ml_project")

def meteostat_extract_data(client, cfg, directory_paths_dict):

    logger.info("Starting to extract MeteoStat weather data.")
    
    outputs_dir = cfg["outputs"]["dir"]
    weather_filename = cfg["outputs"]["files"]["meteostat"]["weather_daily"]
    weather_file_loc = directory_paths_dict["root"] / outputs_dir / weather_filename

    file_exists = os.path.isfile(weather_file_loc)

    if file_exists:
        logger.info("Weather file already exists at %s. Skipping extraction.", weather_file_loc)
        return

    with open(weather_file_loc, "a", encoding="utf-8", newline="") as weather_f:
        rows_written = 0
        for data in client.extract_daily_data(cfg):
            if write_csv(weather_f, data):
                rows_written += 1
        
        if rows_written > 0:
            logger.debug("Successfully wrote %d rows to %s.", rows_written, weather_file_loc)
        else:
            logger.warning("No weather data written to %s.", weather_file_loc)