import requests

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
        query_params = {
            "lat": cfg["data"]["coordinates"]["latitude"],
            "lon": cfg["data"]["coordinates"]["longitude"],
            "start": cfg["time"]["start"].strftime("%Y-%m-%d"),
            "end": cfg["time"]["end"].strftime("%Y-%m-%d"),
            "units": cfg["sources"]["meteostat"]["units"],
            "model": cfg["sources"]["meteostat"]["model"]
        }

        try:
            response = requests.get(url, headers=self.headers, params=query_params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            raise e

        yield from payload["data"]


