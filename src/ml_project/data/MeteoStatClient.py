import requests
import logging

logger = logging.getLogger("ml_project")

class MeteoStatClient():

    def __init__(self, api_key: str, base_url: str = "meteostat.p.rapidapi.com", timeout: int = 10):
        self.api_key = api_key
        self.base_url = "https://meteostat.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": base_url
        }
        self.timeout = timeout

    def extract_daily_data(self, cfg):
        url = f"{self.base_url}/point/daily"
        for coord in cfg["data"]["coordinates"]:
            query_params = {
                "lat": coord[0],
                "lon": coord[1],
                "start": cfg["time"]["start"].strftime("%Y-%m-%d"),
                "end": cfg["time"]["end"].strftime("%Y-%m-%d"),
                "units": cfg["sources"]["meteostat"]["units"],
                "model": cfg["sources"]["meteostat"]["model"]
            }

            try:
                logger.info("Requesting daily data from MeteoStat.")
                response = requests.get(url, headers=self.headers, params=query_params, timeout=self.timeout)
                response.raise_for_status()
                weather_daily_data = response.json()["data"]
            except Exception as e:
                logger.error("Error when requesting MeteoStat daily data: %s", e)
                raise e
            
            for record in weather_daily_data:
                record["latitude"] = coord[0]
                record["longitude"] = coord[1]

            yield from weather_daily_data


