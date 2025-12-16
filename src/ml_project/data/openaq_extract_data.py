from typing import Dict, Iterable, List, Tuple
from datetime import datetime, timezone

def iter_locations(client, open_aq_cfg: Dict) -> Iterable[Tuple[dict, List[dict]]]:
    """Yield location row dicts that satisfy the configured date filter."""
    page = 1
    while True:
        location_response = client.locations.list(
            coordinates=(
                open_aq_cfg["coordinates"]["latitude"],
                open_aq_cfg["coordinates"]["longitude"],
            ),
            radius=open_aq_cfg["radius"],
            limit=open_aq_cfg["limit"],
            page=page,
        )

        # Exists when there are no longer any results from pagination
        if not location_response.results:
            break

        for l in location_response.results:
            location_rows = {
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

            last_read = datetime.fromisoformat(location_rows["last_read_at"])
            min_dt = open_aq_cfg["daterange"]["min"].replace(tzinfo=timezone.utc)

            if last_read >= min_dt:
                sensor_list = _extract_sensors_from_location(l)
                yield location_rows, sensor_list

        page += 1

def iter_sensor_measurements(client, open_aq_cfg: Dict, sensor_id: int) -> Iterable[dict]:
    """Yield measurement rows for a sensor across all pages."""
    page = 1

    while True:
        sensor_data_response = client.measurements.list(
            sensors_id=sensor_id,
            datetime_from=open_aq_cfg["daterange"]["min"],
            datetime_to=open_aq_cfg["daterange"]["max"],
            limit=open_aq_cfg["limit"],
            rollup=open_aq_cfg["rollup"],
            page=page,
        )
        print(f"Sensor ID: {sensor_id} at page {page} with results length of {len(sensor_data_response.results)}")

        # Exists when there are no longer any results from pagination
        if not sensor_data_response.results:
            break

        for sd in sensor_data_response.results:
            sensor_measurement_row = {
                "sensor_id": sensor_id,
                "datetime_from": sd.period.datetime_from.utc,
                "datetime_to": sd.period.datetime_to.utc,
                "timestamp_rollup": open_aq_cfg["rollup"],
                "value": sd.value,
                "metric_name": sd.parameter.name,
                "units": sd.parameter.units,
            }

            yield sensor_measurement_row

        page += 1

def _extract_sensors_from_location(location_result) -> List[dict]:
    """Extract sensors metadata rows from a single location result object."""
    sensor_location_list = []
    for s in location_result.sensors:
        sensor_location_list.append({
                "sensor_id": s.id,
                "measurement": s.parameter.display_name,
                "measurement_name": s.parameter.name,
                "units": s.parameter.units,
                "location_id": location_result.id,
        })

    return sensor_location_list