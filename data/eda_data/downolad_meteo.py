import time
import os
import requests
import numpy as np
import pandas as pd

LATITUDE = 51.1155   # stacja GIOŚ wyb. Conrada-Korzeniowskiego
LONGITUDE = 17.0283

START_DATE = "2020-01-01"
END_DATE = "2026-03-31"

OUTPUT_CSV = "./data/openmeteo_wroclaw_forecast_daily.csv"

# ECMWF IFS HRES — 9 km, dane od 2017, godzinowy
WEATHER_MODEL = "ecmwf_ifs"

DAILY_VARIABLES = [
    # Temperatura (Czernecki: T2_avg, T2_min, T2_max)
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    # Wilgotność (Czernecki: brak bezpośrednio, ale Kaya & Bucak używają)
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
    # Wiatr (Czernecki: WS_avg, WS_max, GUST)
    "windspeed_10m_max",
    "windspeed_10m_mean",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
    # Ciśnienie (Czernecki: SLP)
    "pressure_msl_mean",
    # Opady (Czernecki: PREC)
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    # Nasłonecznienie (Czernecki: SUN; Kaya: solar irradiance)
    "shortwave_radiation_sum",
    "sunshine_duration",
    # Zachmurzenie (Czernecki: TOT_CL)
    "cloudcover_mean",
    # Ewapotranspiracja
    "et0_fao_evapotranspiration",
]

HOURLY_VARIABLES = [
    # Planetary Boundary Layer Height (Czernecki: PBL — najsilniejszy predyktor!)
    "boundary_layer_height",
    # Punkt rosy (Czernecki: DPT_avg, DPT_min, DPT_max)
    "dewpoint_2m",
    # Zachmurzenie niskie (Czernecki: LOW_CL)
    "cloudcover_low",
    # Prędkość wiatru — do obliczenia min (brak w daily API)
    "windspeed_10m",
    # Ciśnienie powierzchniowe (Kaya & Bucak)
    "surface_pressure",
]


BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_chunk(start_date, end_date):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": WEATHER_MODEL,
        "timezone": "Europe/Warsaw",
    }

    resp = requests.get(BASE_URL, params=params, timeout=120)

    if resp.status_code != 200:
        print(f"  Błąd HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    data = resp.json()

    if "error" in data and data["error"]:
        print(f"  Błąd API: {data.get('reason', 'unknown')}")
        return None

    return data


def aggregate_hourly_to_daily(hourly_data):
    df = pd.DataFrame(hourly_data)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date.astype(str)

    agg = {}

    # BLH — mean, min, max (Czernecki: PBL_avg, PBL_min, PBL_max)
    if "boundary_layer_height" in df.columns:
        agg["blh_mean"] = ("boundary_layer_height", "mean")
        agg["blh_min"] = ("boundary_layer_height", "min")
        agg["blh_max"] = ("boundary_layer_height", "max")

    # Punkt rosy — mean, min, max (Czernecki: DPT_avg, DPT_min, DPT_max)
    if "dewpoint_2m" in df.columns:
        agg["dewpoint_2m_mean"] = ("dewpoint_2m", "mean")
        agg["dewpoint_2m_min"] = ("dewpoint_2m", "min")
        agg["dewpoint_2m_max"] = ("dewpoint_2m", "max")

    # Zachmurzenie niskie — mean (Czernecki: LOW_CL)
    if "cloudcover_low" in df.columns:
        agg["cloudcover_low_mean"] = ("cloudcover_low", "mean")

    # Prędkość wiatru — min (brak w daily API)
    if "windspeed_10m" in df.columns:
        agg["windspeed_10m_min"] = ("windspeed_10m", "min")

    # Ciśnienie powierzchniowe — mean (Kaya & Bucak)
    if "surface_pressure" in df.columns:
        agg["surface_pressure_mean"] = ("surface_pressure", "mean")

    if not agg:
        return pd.DataFrame()

    result = df.groupby("date").agg(**agg).round(2).reset_index()
    return result


def main():
    print("Open-Meteo Historical Forecast API — Wrocław")
    print(f"Model:          {WEATHER_MODEL} (ECMWF IFS HRES, 9 km)")
    print(f"Zakres:         {START_DATE} → {END_DATE}")
    print(f"Współrzędne:    {LATITUDE}°N, {LONGITUDE}°E")
    print(f"Zmienne daily:  {len(DAILY_VARIABLES)}")
    print(f"Zmienne hourly: {len(HOURLY_VARIABLES)} (→ agregowane do dziennych)")
    print()

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    all_daily = []
    all_hourly_agg = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(
            chunk_start + pd.DateOffset(months=3) - pd.Timedelta(days=1), end
        )
        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")

        print(f"Pobieram {s} → {e}...", end=" ", flush=True)

        data = fetch_chunk(s, e)

        if data:
            if "daily" in data:
                df_d = pd.DataFrame(data["daily"])
                df_d.rename(columns={"time": "date"}, inplace=True)
                all_daily.append(df_d)

            if "hourly" in data:
                df_h = aggregate_hourly_to_daily(data["hourly"])
                if not df_h.empty:
                    all_hourly_agg.append(df_h)

            n_days = len(df_d) if "daily" in data else 0
            print(f"({n_days} dni)")

            if len(all_daily) == 1:
                elev = data.get("elevation", "?")
                print(f"Elewacja modelu: {elev} m")
        else:
            print("brak danych")

        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(1)

    if not all_daily:
        print("\nNie udało się pobrać żadnych danych!")
        return

    result = pd.concat(all_daily, ignore_index=True)
    result = result.sort_values("date").drop_duplicates(subset="date")

    if all_hourly_agg:
        hourly_agg = pd.concat(all_hourly_agg, ignore_index=True)
        hourly_agg = hourly_agg.sort_values("date").drop_duplicates(subset="date")
        result = result.merge(hourly_agg, on="date", how="left")

    result.to_csv(OUTPUT_CSV, index=False)

    print(f"Plik:    {OUTPUT_CSV}")
    print(f"Wierszy: {len(result)}")
    print(f"Zakres:  {result['date'].min()} → {result['date'].max()}")
    print(f"Kolumn:  {len(result.columns)}")


if __name__ == "__main__":
    main()