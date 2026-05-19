import pandas as pd
import numpy as np
import re
import os
import glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(_SCRIPT_DIR, "gios")
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "gios_daily_agg.csv")
FILE_PATTERN = "*.csv"

TARGET_POLLUTANTS = ["PM10", "PM25", "SO2", "CO", "NO2", "O3"]

STATION_MAP = {
    "Al. Wiśniowa": "Wisniowa",
    "ul. Orzechowa": "Orzechowa",
    "wyb. Conrada-Korzeniowskiego": "Conrada",
    "ul. Bartnicza": "Bartnicza",
    "ul. Na Grobli": "NaGrobli",
}

FALLBACK_ORDER = {
    "PM10": ["Conrada", "Orzechowa", "NaGrobli", "Wisniowa", "Bartnicza"],
    "PM25": ["Conrada", "Wisniowa", "Orzechowa", "NaGrobli", "Bartnicza"],
    "SO2":  ["Conrada", "Orzechowa", "NaGrobli", "Wisniowa", "Bartnicza"],
    "CO":   ["Conrada", "Wisniowa", "Orzechowa", "NaGrobli", "Bartnicza"],
    "NO2":  ["Conrada", "Wisniowa", "Bartnicza", "Orzechowa", "NaGrobli"],
    "O3":   ["Conrada", "Bartnicza", "Wisniowa", "Orzechowa", "NaGrobli"],
}

POLLUTANT_MAP = {
    "pył zawieszony PM10": "PM10",
    "pył zawieszony PM2.5": "PM25",
    "dwutlenek siarki": "SO2",
    "dwutlenek azotu": "NO2",
    "tlenek węgla": "CO",
    "ozon": "O3",
    "benzen": "C6H6",
    "benzo(a)piren": "BaP",
    "arsen": "As",
    "kadm": "Cd",
    "nikiel": "Ni",
    "ołów": "Pb",
}


def parse_column_name(raw):
    raw = raw.strip()
    if raw.lower() in ("data", ""):
        return None

    station_match = re.search(r'Wrocław\s*-\s*(.+?)\s*\(', raw)
    if not station_match:
        return None
    station_raw = station_match.group(1).strip()
    station = next((v for k, v in STATION_MAP.items() if k in station_raw), station_raw)

    poll_match = re.search(r'\((.+?)\s*\[', raw)
    if not poll_match:
        return None
    poll_raw = poll_match.group(1).strip()
    pollutant = next((v for k, v in POLLUTANT_MAP.items() if k in poll_raw), poll_raw)

    return (station, pollutant)


def _classify_auto_vs_grav(series):
    """
    GIOŚ publishes two PM10 columns for the same station:
    - automatic: 22-24 valid readings per day
    - gravimetric: 1 reading per day at midnight

   median readings-per-day: automatic columns have >>1.
    """
    if series.dropna().empty:
        return "unknown"
    daily_counts = series.dropna().groupby(series.dropna().index.normalize()).count()
    median_count = daily_counts.median()
    return "auto" if median_count > 3 else "grav"


def read_gios_csv(filepath):
    with open(filepath, encoding='utf-8-sig') as f:
        lines = f.readlines()

    header_raw = lines[0].strip().replace('\r', '').split('*')

    col_names = []
    seen = {}
    for h in header_raw:
        parsed = parse_column_name(h)
        if parsed is None:
            col_names.append("datetime")
        else:
            name = f"{parsed[0]}_{parsed[1]}"
            if name in seen:
                seen[name] += 1
                name = f"{name}__dup{seen[name]}"
            else:
                seen[name] = 0
            col_names.append(name)

    n_cols = len(col_names)
    data_rows = []
    for line in lines[1:]:
        line = line.strip().replace('\r', '')
        if line:
            data_rows.append(line.split(',')[:n_cols])

    df = pd.DataFrame(data_rows, columns=col_names)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M', errors='coerce')

    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')

    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime')

    # --- FIX: merge duplicate columns instead of dropping them ---
    # For each group of duplicates (e.g. Conrada_PM10 and Conrada_PM10__dup1),
    # identify which is automatic (hourly) and which is gravimetric (daily).
    # Keep the automatic column under the canonical name.
    # Store the gravimetric column as Station_PM10_grav for potential fallback.

    base_names = {}  
    for col in df.columns:
        canonical = col.split('__dup')[0]
        base_names.setdefault(canonical, []).append(col)

    cols_to_drop = []
    cols_to_rename = {}

    for canonical, variants in base_names.items():
        if len(variants) == 1:
            continue  

        classified = []
        for v in variants:
            typ = _classify_auto_vs_grav(df[v])
            classified.append((v, typ))

        auto_cols = [v for v, t in classified if t == "auto"]
        grav_cols = [v for v, t in classified if t == "grav"]

        if auto_cols:
            keeper = auto_cols[0]
            if keeper != canonical:
                cols_to_rename[keeper] = canonical
                if canonical in df.columns:
                    cols_to_rename[canonical] = f"{canonical}_grav"
            for v in auto_cols[1:]:
                cols_to_drop.append(v)
            for i, v in enumerate(grav_cols):
                if v not in cols_to_rename and v != canonical:
                    cols_to_drop.append(v) 
        else:
            for v in variants[1:]:
                cols_to_drop.append(v)

    if cols_to_rename:
        df = df.rename(columns=cols_to_rename)
    if cols_to_drop:
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df = df.reset_index()
    return df


def merge_monthly_files(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    print(f"Found {len(files)} CSV files")

    all_dfs = []
    for filepath in files:
        try:
            df = read_gios_csv(filepath)
            fname = os.path.basename(filepath)
            print(f"  {fname}: {len(df)} rows ({df['datetime'].min().date()} -> {df['datetime'].max().date()})")
            all_dfs.append(df)
        except Exception as e:
            print(f"  ERROR {os.path.basename(filepath)}: {e}")

    if not all_dfs:
        raise ValueError("No files were successfully read!")

    merged = pd.concat(all_dfs, ignore_index=True)
    return merged.sort_values('datetime').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Daily aggregation rules per Directive 2008/50/EC, Annex XI
# (transposed into Polish law: Rozp. MŚ z 24.08.2012, Dz.U. 2021 poz. 845)
#
#   PM10, PM2.5  -> 24-hour mean  (averaging period: 1 day)
#   SO2          -> max 1-hour    (hourly limit value: 350 µg/m³)
#   NO2          -> max 1-hour    (hourly limit value: 200 µg/m³)
#   O3           -> max daily 8-h running mean (target value: 120 µg/m³)
#   CO           -> max daily 8-h running mean (limit value: 10 mg/m³)
#
# Minimum data capture (per EEA AirBase aggregation rules):
#   - mean24h, max1h: ≥18 valid hourly values per day (75% of 24 h)
#   - max8h: each 8-h running mean requires ≥6 valid hourly values (75% of 8 h),
#            and the daily max requires ≥18 valid 8-h running means per day
# ---------------------------------------------------------------------------

DAILY_AGG = {
    "PM10": "mean24h",
    "PM25": "mean24h",
    "NO2":  "max1h",
    "SO2":  "max1h",
    "O3":   "max8h",
    "CO":   "max8h",
}

MIN_HOURS = 18  # 75% of 24 hours — applies to mean24h, max1h, and number of valid 8h-windows for max8h
MIN_HOURS_IN_8H_WINDOW = 6  # 75% of 8 hours — applies to validity of each individual 8h running mean


def _agg_column(series_hourly, agg_type):
    if agg_type == "mean24h":
        def daily_mean(x):
            valid = x.dropna()
            return valid.mean() if len(valid) >= MIN_HOURS else np.nan
        return series_hourly.resample('D').apply(daily_mean)

    if agg_type == "max1h":
        def daily_max(x):
            valid = x.dropna()
            return valid.max() if len(valid) >= MIN_HOURS else np.nan
        return series_hourly.resample('D').apply(daily_max)

    if agg_type == "max8h":
        rolling8h = series_hourly.rolling(window=8, min_periods=MIN_HOURS_IN_8H_WINDOW).mean()
        daily_max = rolling8h.resample('D').max()
        # EEA: a daily 8-hour maximum requires ≥18 valid running 8-hour averages per day
        valid_windows_per_day = rolling8h.notna().resample('D').sum()
        daily_max[valid_windows_per_day < MIN_HOURS] = np.nan
        return daily_max

    raise ValueError(f"Unknown agg_type: {agg_type}")


def build_daily_with_fallback(hourly):
    hourly = hourly.set_index('datetime').sort_index()
    hourly.index = hourly.index.tz_localize(None)

    all_dates = pd.date_range(hourly.index.normalize().min(),
                              hourly.index.normalize().max(), freq='D')

    daily_by_col = {}
    for col in hourly.columns:
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in DAILY_AGG:
            agg_type = DAILY_AGG[parts[1]]
            daily_by_col[col] = _agg_column(hourly[col], agg_type).reindex(all_dates)

    daily_all = pd.DataFrame(daily_by_col, index=all_dates)
    daily_all.index.name = 'date'

    daily_final = pd.DataFrame(index=all_dates)
    daily_final.index.name = 'date'

    for pollutant in TARGET_POLLUTANTS:
        fallback_stations = FALLBACK_ORDER.get(pollutant, ["Conrada"])
        values = pd.Series(np.nan, index=all_dates, dtype=float)

        for station in fallback_stations:
            col_name = f"{station}_{pollutant}"
            if col_name in daily_all.columns:
                mask = values.isna() & daily_all[col_name].notna()
                values[mask] = daily_all.loc[mask, col_name]

        daily_final[pollutant] = values.round(2)

    return daily_final


def load_existing(path):
    if not os.path.exists(path):
        return None
    try:
        existing = pd.read_csv(path, index_col='date', parse_dates=True)
        print(f"Existing data: {len(existing)} days ({existing.index.min().date()} -> {existing.index.max().date()})")
        return existing
    except Exception:
        return None


if __name__ == "__main__":
    hourly = merge_monthly_files(INPUT_DIR, FILE_PATTERN)
    new_daily = build_daily_with_fallback(hourly)

    existing = load_existing(OUTPUT_PATH)
    if existing is not None:
        combined = existing.combine_first(new_daily)
        combined.update(new_daily)
        daily = combined.sort_index()
        print(f"Merged with existing: {len(existing)} old + {len(daily) - len(existing)} new days")
    else:
        daily = new_daily.sort_index()

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    daily.to_csv(OUTPUT_PATH)

    print(f"\nResult: {len(daily)} days ({daily.index.min().date()} -> {daily.index.max().date()})")
    for col in TARGET_POLLUTANTS:
        valid = daily[col].notna().sum()
        print(f"  {col}: {valid}/{len(daily)} ({100*valid/len(daily):.0f}%)")
    print(f"Saved: {OUTPUT_PATH}")