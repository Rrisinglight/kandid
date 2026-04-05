#!/usr/bin/env python3
"""
Независимый скрипт: скачивает архивную погоду Open-Meteo за апрель–октябрь
2024 и 2025, фильтрует дни по погодному коду (исключает плохую погоду),
делает сбалансированную выборку по ветровым бинам и сохраняет результат в JSON
и Markdown-отчёт.

Никакой связи с data.py или другими файлами проекта.

Запуск:
    python select_dates_balanced.py [--days-per-bin 13] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CENTER_LAT = 55.915966
CENTER_LON = 36.252330
LOCAL_TIMEZONE = "Europe/Moscow"

PERIODS = [
    ("2024-04-01", "2024-10-31"),
    ("2025-04-01", "2025-10-31"),
]

N_BINS = 5  # количество ветровых бинов

WMO_TEXT_RU: dict[int, str] = {
    0: "Ясно",
    1: "Преимущественно ясно",
    2: "Переменная облачность",
    3: "Пасмурно",
    45: "Туман",
    48: "Туман с изморозью",
    51: "Морось слабая",
    53: "Морось умеренная",
    55: "Морось сильная",
    56: "Морось ледяная слабая",
    57: "Морось ледяная сильная",
    61: "Дождь слабый",
    63: "Дождь умеренный",
    65: "Дождь сильный",
    66: "Ледяной дождь слабый",
    67: "Ледяной дождь сильный",
    71: "Снег слабый",
    73: "Снег умеренный",
    75: "Снег сильный",
    77: "Снежные зёрна",
    80: "Ливень слабый",
    81: "Ливень умеренный",
    82: "Ливень сильный",
    85: "Снегопад слабый",
    86: "Снегопад сильный",
    95: "Гроза",
    96: "Гроза с мелким градом",
    99: "Гроза с крупным градом",
}

BAD_WEATHER_CODES: set[int] = {
    56, 57,
    61, 63, 65, 66, 67,
    71, 73, 75, 77,
    80, 81, 82,
    85, 86,
    95, 96, 99,
}

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "balanced_selection"


# ---------------------------------------------------------------------------
# Bin computation (equal-frequency / quantile-based)
# ---------------------------------------------------------------------------

def compute_bins(
    winds: list[float],
    n_bins: int = N_BINS,
) -> list[tuple[float, float]]:
    """
    Делит все значения ветра на n_bins равных по числу дней частей (квантили).
    Границы округляются до 0.1 м/с. Крайние значения не обрезаются.
    """
    if not winds:
        raise ValueError("Нет данных о ветре для вычисления бинов.")

    s = sorted(winds)
    # n_bins + 1 граничных точек: от min до max
    edges: list[float] = []
    for i in range(n_bins + 1):
        idx = int(round(i * (len(s) - 1) / n_bins))
        edges.append(s[idx])

    edges = sorted(set(round(e, 1) for e in edges))

    bins: list[tuple[float, float]] = []
    for i in range(len(edges) - 1):
        lo_v = edges[i]
        hi_v = edges[i + 1]
        if lo_v < hi_v:
            bins.append((lo_v, hi_v))

    return bins


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def fetch_period(start_date: str, end_date: str) -> dict:
    """Запрашивает почасовые и суточные данные за указанный период."""
    params = {
        "latitude": f"{CENTER_LAT:.6f}",
        "longitude": f"{CENTER_LON:.6f}",
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "cloud_cover",
            "weather_code",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ]),
        "daily": "sunrise,sunset",
        "timezone": LOCAL_TIMEZONE,
        "wind_speed_unit": "ms",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "select-dates-balanced/1.0"})

    for attempt in range(1, 4):
        try:
            with urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            break
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"  Попытка {attempt}/3 не удалась: {exc}")
            if attempt == 3:
                raise RuntimeError(f"Не удалось получить данные: {exc}") from exc
            time.sleep(attempt * 2)

    payload = json.loads(raw)
    if "hourly" not in payload or "daily" not in payload:
        raise RuntimeError(f"Некорректный ответ API: {str(payload)[:300]}")
    return payload


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def parse_hhmm(dt_str: str) -> int:
    """Возвращает час из строки ISO-формата 'YYYY-MM-DDTHH:MM'."""
    return int(dt_str[11:13])


def circular_mean_degrees(angles: list[float]) -> float:
    """Среднее для угловых значений (направление ветра)."""
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    mean_rad = math.atan2(sin_sum, cos_sum)
    return round((math.degrees(mean_rad) + 360) % 360, 1)


def build_daily_records(payload: dict) -> list[dict]:
    """
    Объединяет почасовые и суточные данные в список дневных записей.
    Дневное окно = от часа рассвета до часа заката включительно.
    """
    hourly = payload["hourly"]
    daily_data = payload["daily"]

    # Рассвет/закат по дате: час (для фильтрации окна) и строка HH:MM (для JSON)
    sunrise_hour: dict[str, int] = {}
    sunset_hour: dict[str, int] = {}
    sunrise_str: dict[str, str] = {}
    sunset_str: dict[str, str] = {}
    for i, d in enumerate(daily_data["time"]):
        sr_raw = daily_data["sunrise"][i]
        ss_raw = daily_data["sunset"][i]
        sunrise_hour[d] = parse_hhmm(sr_raw)
        sunset_hour[d] = parse_hhmm(ss_raw)
        sunrise_str[d] = sr_raw[11:16]
        sunset_str[d] = ss_raw[11:16]

    # Группируем почасовые данные по дате
    hourly_by_date: dict[str, list[dict]] = {}
    for i, t in enumerate(hourly["time"]):
        day_str = t[:10]
        hour = int(t[11:13])
        hourly_by_date.setdefault(day_str, []).append({
            "hour": hour,
            "temperature_2m": hourly["temperature_2m"][i],
            "cloud_cover": hourly["cloud_cover"][i],
            "weather_code": hourly["weather_code"][i],
            "wind_speed_10m": hourly["wind_speed_10m"][i],
            "wind_gusts_10m": hourly["wind_gusts_10m"][i],
            "wind_direction_10m": hourly["wind_direction_10m"][i],
        })

    records: list[dict] = []
    for day_str in sorted(hourly_by_date):
        sr = sunrise_hour.get(day_str)
        ss = sunset_hour.get(day_str)
        if sr is None or ss is None:
            continue

        # Часы в дневном окне рассвет..закат (включительно)
        window = [
            h for h in hourly_by_date[day_str]
            if sr <= h["hour"] <= ss
        ]
        if not window:
            continue

        codes = [h["weather_code"] for h in window]
        winds = [h["wind_speed_10m"] for h in window if h["wind_speed_10m"] is not None]
        gusts = [h["wind_gusts_10m"] for h in window if h["wind_gusts_10m"] is not None]
        temps = [h["temperature_2m"] for h in window if h["temperature_2m"] is not None]
        clouds = [h["cloud_cover"] for h in window if h["cloud_cover"] is not None]
        dirs = [h["wind_direction_10m"] for h in window if h["wind_direction_10m"] is not None]

        worst_code = max(codes)

        records.append({
            "date": day_str,
            "sunrise": sunrise_str[day_str],
            "sunset": sunset_str[day_str],
            "has_bad_weather": any(c in BAD_WEATHER_CODES for c in codes),
            "weather_code": worst_code,
            "weather_code_text": WMO_TEXT_RU.get(worst_code, str(worst_code)),
            "temperature_2m_mean": round(statistics.mean(temps), 1) if temps else None,
            "wind_speed_10m_mean": round(statistics.mean(winds), 2) if winds else None,
            "wind_gusts_10m_max": round(max(gusts), 2) if gusts else None,
            "wind_direction_10m_mean": circular_mean_degrees(dirs),
            "cloud_cover_mean": round(statistics.mean(clouds), 1) if clouds else None,
        })

    return records


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def even_sample(items: list, n: int) -> list:
    """Равномерная шаг-выборка n элементов из отсортированного списка."""
    if n >= len(items):
        return items[:]
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def balanced_sample(
    records: list[dict],
    wind_bins: list[tuple[float, float]],
    days_per_bin: int,
) -> list[dict]:
    """Сбалансированная выборка по ветровым бинам из отфильтрованных дней."""
    eligible = [r for r in records if not r["has_bad_weather"]]
    selected: list[dict] = []
    for lo, hi in wind_bins:
        bucket = [
            r for r in eligible
            if r["wind_speed_10m_mean"] is not None
            and lo <= r["wind_speed_10m_mean"] < hi
        ]
        bucket_sorted = sorted(bucket, key=lambda r: r["date"])
        chosen = even_sample(bucket_sorted, days_per_bin)
        selected.extend(chosen)
    return sorted(selected, key=lambda r: r["date"])


# ---------------------------------------------------------------------------
# Output: clean JSON (drop internal flag)
# ---------------------------------------------------------------------------

def to_output_record(r: dict) -> dict:
    """Убирает служебное поле has_bad_weather перед сохранением."""
    return {k: v for k, v in r.items() if k != "has_bad_weather"}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_markdown(
    all_records: list[dict],
    selected: list[dict],
    wind_bins: list[tuple[float, float]],
    start_1: str,
    end_2: str,
) -> str:
    eligible = [r for r in all_records if not r["has_bad_weather"]]

    lines = [
        "# Анализ реальной погоды",
        "",
        f"- Период: {start_1} - {end_2}",
        f"- Дней в периоде: {len(all_records)}",
        f"- Дней без плохой погоды в окне рассвет–закат: {len(eligible)}",
        "",
        "## Рекомендованная сбалансированная выборка",
        "",
        f"- Количество рекомендованных дат: {len(selected)}",
    ]
    if selected:
        lines.extend([
            f"- Период рекомендованных дат: {selected[0]['date']} - {selected[-1]['date']}",
            f"- Средний ветер по рекомендованным датам: "
            f"{statistics.mean(r['wind_speed_10m_mean'] for r in selected if r['wind_speed_10m_mean']):.2f} м/с",
            f"- Средние порывы по рекомендованным датам: "
            f"{statistics.mean(r['wind_gusts_10m_max'] for r in selected if r['wind_gusts_10m_max']):.2f} м/с",
        ])
    lines.append("")
    lines.append("## Распределение по ветровым бинам")
    lines.append("")
    for lo, hi in wind_bins:
        count = sum(
            1 for r in eligible
            if r["wind_speed_10m_mean"] is not None
            and lo <= r["wind_speed_10m_mean"] < hi
        )
        lines.append(f"- [{lo:.1f}, {hi:.1f}) м/с: {count} дней")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сбалансированная выборка погодных дней 2024–2025 без зависимостей от data.py"
    )
    parser.add_argument(
        "--days-per-bin", type=int, default=13,
        help="Кол-во дней из каждого ветрового бина (по умолч. 13 → ~78 дней)",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Папка для сохранения JSON и Markdown-отчёта",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for start_date, end_date in PERIODS:
        print(f"Скачиваем данные {start_date} – {end_date}...")
        payload = fetch_period(start_date, end_date)
        records = build_daily_records(payload)
        print(f"  Получено дней: {len(records)}")
        all_records.extend(records)

    all_records.sort(key=lambda r: r["date"])
    print(f"\nВсего дней в двух периодах: {len(all_records)}")

    eligible = [r for r in all_records if not r["has_bad_weather"]]
    print(f"Дней без плохой погоды: {len(eligible)}")

    # Вычисляем бины из реальных данных
    eligible_winds = [
        r["wind_speed_10m_mean"] for r in eligible if r["wind_speed_10m_mean"] is not None
    ]
    wind_bins = compute_bins(eligible_winds, n_bins=N_BINS)
    print(f"\nАвтоматические бины (квантильное деление на {N_BINS} равных частей):")
    for lo, hi in wind_bins:
        count = sum(
            1 for r in eligible
            if r["wind_speed_10m_mean"] is not None
            and lo <= r["wind_speed_10m_mean"] < hi
        )
        print(f"  [{lo:.1f}–{hi:.1f}) м/с: {count} дней")

    selected = balanced_sample(all_records, wind_bins, args.days_per_bin)
    print(f"\nВыбрано дней (сбалансированно): {len(selected)}")

    print(f"\nРаспределение по бинам (после выборки):")
    for lo, hi in wind_bins:
        count = sum(
            1 for r in selected
            if r["wind_speed_10m_mean"] is not None
            and lo <= r["wind_speed_10m_mean"] < hi
        )
        print(f"  [{lo:.1f}–{hi:.1f}) м/с: {count} дней")

    # JSON
    json_path = output_dir / "selected_days_2024-2025.json"
    json_path.write_text(
        json.dumps([to_output_record(r) for r in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON сохранён: {json_path}")

    # Markdown
    period_start = PERIODS[0][0]
    period_end = PERIODS[-1][1]
    md_text = build_markdown(all_records, selected, wind_bins, period_start, period_end)
    md_path = output_dir / "weather_report_2024-2025.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Markdown сохранён: {md_path}")


if __name__ == "__main__":
    main()
