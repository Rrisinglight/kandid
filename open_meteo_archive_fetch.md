# Historical Weather API — hourly data for one day or a date range

Use the public **Historical Weather API** when you need archived (reanalysis-based) time series for a point location. This note covers **HTTP query parameters** and **Python** usage only. Self-hosted Open-Meteo, local datasets, and model downloaders are out of scope.

**Endpoint**

```
https://archive-api.open-meteo.com/v1/archive
```

Full parameter reference and dataset coverage: [Historical Weather API documentation](https://open-meteo.com/en/docs/historical-weather-api).

---

## Request

**To obtain hourly fields for a location and calendar interval**

1. Set `latitude` and `longitude` (decimal degrees).
2. Set `start_date` and `end_date` as `YYYY-MM-DD`. The interval is **inclusive** on both ends.
3. Set `hourly` to a comma-separated list of variable names (no spaces). Example: `temperature_2m,wind_speed_10m`.
4. Optionally set `timezone` (e.g. `Europe/Moscow`) so `hourly.time` is aligned to that zone.
5. Optionally set units, e.g. `wind_speed_unit=ms`, `precipitation_unit=mm`.

**Single calendar day:** use the same date for `start_date` and `end_date`.

**Example URL (one day, illustrative variables)**

```text
https://archive-api.open-meteo.com/v1/archive?latitude=55.915966&longitude=36.252330&start_date=2025-06-01&end_date=2025-06-01&hourly=temperature_2m,cloud_cover,weather_code,wind_speed_10m,wind_gusts_10m,precipitation&timezone=Europe%2FMoscow&wind_speed_unit=ms&precipitation_unit=mm
```

**Common query parameters for hourly archives**

| Parameter | Purpose |
|-----------|---------|
| `latitude`, `longitude` | Location |
| `start_date`, `end_date` | Inclusive `YYYY-MM-DD` range |
| `hourly` | Comma-separated hourly variable names |
| `timezone` | IANA timezone for timestamps |
| `wind_speed_unit` | e.g. `ms` for m/s |
| `precipitation_unit` | e.g. `mm` for hourly precipitation |

Variable names must match the API; see the official docs for the full list.

---

## Response

**To read hourly series from JSON**

Parse the top-level object and use the `hourly` object. It contains parallel arrays of equal length:

- `hourly.time` — ISO 8601 timestamps as strings
- One array per variable requested in `hourly=...`, using the same names

If `hourly` is missing or the request failed, handle the error payload from the API before indexing `hourly`.

---

## curl

**To download the response to a file**

```bash
curl -L -G "https://archive-api.open-meteo.com/v1/archive" \
  --data-urlencode "latitude=55.915966" \
  --data-urlencode "longitude=36.252330" \
  --data-urlencode "start_date=2025-04-01" \
  --data-urlencode "end_date=2025-04-30" \
  --data-urlencode "hourly=temperature_2m,cloud_cover,weather_code,wind_speed_10m,wind_gusts_10m,precipitation" \
  --data-urlencode "timezone=Europe/Moscow" \
  --data-urlencode "wind_speed_unit=ms" \
  --data-urlencode "precipitation_unit=mm" \
  -o archive.json
```

---

## Python: one day, standard library only

```python
from datetime import date
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HOURLY = [
    "temperature_2m",
    "cloud_cover",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
]


def fetch_hourly_one_day(lat: float, lon: float, day: date, *, timezone: str) -> dict:
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "hourly": ",".join(HOURLY),
        "timezone": timezone,
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "my-app/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "hourly" not in data:
        raise RuntimeError("missing hourly in response")
    return data
```

---

## Python: inclusive date range with retries

```python
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HOURLY = [
    "temperature_2m",
    "cloud_cover",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
]


def fetch_hourly_range(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    *,
    timezone: str,
    user_agent: str = "my-app/1.0",
    max_attempts: int = 3,
) -> dict:
    params = {
        "latitude": f"{lat:.6f}",
        "longitude": f"{lon:.6f}",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY),
        "timezone": timezone,
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    req = Request(url, headers={"User-Agent": user_agent})
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "hourly" not in data:
                raise RuntimeError("missing hourly in response")
            return data
        except (HTTPError, URLError, TimeoutError) as e:
            if attempt == max_attempts:
                raise RuntimeError(str(e)) from e
            time.sleep(attempt * 2)
```

---

## Python: build a `pandas.DataFrame`

```python
import pandas as pd

def hourly_payload_to_frame(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(h["time"]),
            "temperature_2m": h["temperature_2m"],
            "cloud_cover": h["cloud_cover"],
            "weather_code": h["weather_code"],
            "wind_speed_10m": h["wind_speed_10m"],
            "wind_gusts_10m": h["wind_gusts_10m"],
            "precipitation": h["precipitation"],
        }
    )
```

---

## Daily aggregates instead of hourly

**To obtain built-in daily statistics**

Add `daily` with a comma-separated list of daily variable names (see the official docs). The JSON will include a `daily` block alongside `hourly` if both are requested. You may omit `hourly` and request only `daily` when sub-daily resolution is not required.

```python
params = {
    "latitude": "55.915966",
    "longitude": "36.252330",
    "start_date": "2025-04-01",
    "end_date": "2025-04-30",
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
    "timezone": "Europe/Moscow",
}
```

---

## Notes

- Long `start_date`–`end_date` spans increase payload size; split into smaller ranges if needed.
- Individual hourly values may be `null`; treat as missing in analysis.
- Set an identifiable `User-Agent` on programmatic requests.

For questions and updates to the API, see the [Open-Meteo GitHub Discussions](https://github.com/open-meteo/open-meteo/discussions).
