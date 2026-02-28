import requests
import logging

logger = logging.getLogger("ml_project")

class MeteoStatClient():
    """Client for querying MeteoStat daily point data via RapidAPI.

    Parameters
    - api_key (str): RapidAPI key for authentication.
    - base_url (str): Hostname for the MeteoStat RapidAPI service.
      The full URL scheme is set internally to HTTPS.
    - timeout (int): Request timeout in seconds.

    Attributes
    - headers (dict): Prepared request headers including the API key.
    - timeout (int): Timeout used for HTTP requests.
    """

    def __init__(self, api_key: str, base_url: str = "meteostat.p.rapidapi.com", timeout: int = 10):
        self.api_key = api_key
        self.base_url = "https://meteostat.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": base_url
        }
        self.timeout = timeout

    def extract_daily_data(self, cfg):
        """Yield daily weather records for coordinates described in `cfg`.

        The `cfg` object is expected to be a dictionary with the keys:
        - `data.coordinates`: iterable of (lat, lon) tuples
        - `time.start` / `time.end`: datetime-like objects with `strftime`
        - `sources.meteostat.units` and `sources.meteostat.model` strings

        This method yields dictionaries representing daily measurements
        as returned by the MeteoStat API. Each yielded record is
        annotated with `latitude` and `longitude` keys corresponding to
        the coordinate the record belongs to.

        Yields
        - dict: weather measurement record with added `latitude` and
          `longitude` keys.
        """

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


