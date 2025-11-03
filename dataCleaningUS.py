import warnings
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# ---------- Configuration ----------
INPUT_CSV = "US_Accidents_March23_sampled_500k.csv"
OUTPUT_PROCESSED = "US_Accidents_processed_for_modeling.csv"
DROP_MISSING_COL_THRESHOLD = 0.40  # drop columns with >40% missing
# columns that are noisy / textual and safe to drop before processing
TEXT_DROP_CANDIDATES = ['ID', 'Description', 'Street', 'Weather_Timestamp', 'Source']
# skip these names when looking for datetime-like columns (they contain "time" but are not datetimes)
DATETIME_NAME_SKIP = {'timezone'}

# ---------- Load data ----------
usa_pd = pd.read_csv(INPUT_CSV)
print(f"Loaded dataset: rows={usa_pd.shape[0]:,d}, cols={usa_pd.shape[1]}")

# ---------- Utilities / small helpers ----------
def try_parse_datetime(series, strict_fmt="%Y-%m-%d %H:%M:%S"):
    """Try strict parse first, then fallback to pandas generic parse."""
    parsed = pd.to_datetime(series, format=strict_fmt, errors='coerce')
    # if many failed with strict format, fallback to generic parsing
    if parsed.isna().mean() > 0.05:
        parsed = pd.to_datetime(series, errors='coerce')
    return parsed

# ---------- Data cleaning steps (organized into simple functions) ----------
def drop_text_columns(df, candidates):
    """Drop obvious text/id columns that are unlikely to help models."""
    drop_cols = [c for c in candidates if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped text columns: {drop_cols}")
    return df

def parse_datetimes(df):
    """Parse columns with 'time' or 'date' in name, excluding known non-datetime names."""
    dt_cols = [
        c for c in df.columns
        if ('time' in c.lower() or 'date' in c.lower()) and c.lower() not in DATETIME_NAME_SKIP
    ]
    for c in dt_cols:
        df[c] = try_parse_datetime(df[c])
    print(f"Parsed datetime columns: {dt_cols}")
    return df

def drop_sparse_columns(df, threshold=DROP_MISSING_COL_THRESHOLD):
    """Drop columns with fraction missing greater than threshold."""
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Dropped {len(to_drop)} columns with >{int(threshold*100)}% missing: {to_drop}")
    else:
        print("No columns exceeded missingness threshold.")
    return df, to_drop

def drop_rows_with_any_na(df):
    """Drop rows that contain any NA (after column drops). Returns df and counts."""
    before = df.shape[0]
    mask_keep = ~df.isna().any(axis=1)
    df = df.loc[mask_keep].reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped rows with any NA: before={before:,d}, after={after:,d}, dropped={before-after:,d}")
    return df

def add_time_features(df):
    """Create simple time-based features from Start_Time/End_Time if available."""
    if 'Start_Time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Start_Time']):
        df['start_hour'] = df['Start_Time'].dt.hour
        df['start_dayofweek'] = df['Start_Time'].dt.dayofweek
        df['start_month'] = df['Start_Time'].dt.month
    if set(['Start_Time', 'End_Time']).issubset(df.columns) and \
       pd.api.types.is_datetime64_any_dtype(df['Start_Time']) and \
       pd.api.types.is_datetime64_any_dtype(df['End_Time']):
        df['duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        df['duration_min'] = df['duration_min'].clip(lower=0)
    return df

def convert_imperial_to_metric(df):
    """Convert commonly used imperial columns to metric equivalents and drop originals."""
    # Temperature F -> Celsius
    if 'Temperature(F)' in df.columns:
        df['Temperature(C)'] = pd.to_numeric(df['Temperature(F)'], errors='coerce').apply(
            lambda x: (x - 32) * 5.0/9.0 if pd.notna(x) else x
        )
        df.drop(columns=['Temperature(F)'], inplace=True)

    # Wind Chill F -> Celsius
    if 'Wind_Chill(F)' in df.columns:
        df['Wind_Chill(C)'] = pd.to_numeric(df['Wind_Chill(F)'], errors='coerce').apply(
            lambda x: (x - 32) * 5.0/9.0 if pd.notna(x) else x
        )
        df.drop(columns=['Wind_Chill(F)'], inplace=True)

    # Distance miles -> kilometers
    if 'Distance(mi)' in df.columns:
        df['Distance(km)'] = pd.to_numeric(df['Distance(mi)'], errors='coerce') * 1.609344
        df.drop(columns=['Distance(mi)'], inplace=True)

    # Wind speed mph -> m/s
    if 'Wind_Speed(mph)' in df.columns:
        df['Wind_Speed(m/s)'] = pd.to_numeric(df['Wind_Speed(mph)'], errors='coerce') * 0.44704
        df.drop(columns=['Wind_Speed(mph)'], inplace=True)

    # Visibility miles -> kilometers
    if 'Visibility(mi)' in df.columns:
        df['Visibility(km)'] = pd.to_numeric(df['Visibility(mi)'], errors='coerce') * 1.609344
        df.drop(columns=['Visibility(mi)'], inplace=True)

    # Precipitation inches -> millimeters
    if 'Precipitation(in)' in df.columns:
        df['Precipitation(mm)'] = pd.to_numeric(df['Precipitation(in)'], errors='coerce') * 25.4
        df.drop(columns=['Precipitation(in)'], inplace=True)

    # Pressure inches Hg -> hPa
    if 'Pressure(in)' in df.columns:
        df['Pressure(hPa)'] = pd.to_numeric(df['Pressure(in)'], errors='coerce') * 33.8638866667
        df.drop(columns=['Pressure(in)'], inplace=True)

    return df

# ---------- Pipeline (simple linear sequence) ----------
# 1) Drop noisy text/id columns
usa_pd = drop_text_columns(usa_pd, TEXT_DROP_CANDIDATES)

# 2) Parse datetime-like columns (safe fallback parsing implemented)
usa_pd = parse_datetimes(usa_pd)

# 3) Drop high-missing columns
usa_pd, dropped_cols = drop_sparse_columns(usa_pd, DROP_MISSING_COL_THRESHOLD)

# 4) Drop any remaining rows with NA (user choice)
usa_pd = drop_rows_with_any_na(usa_pd)

# 5) Feature engineering: simple time features and duration
usa_pd = add_time_features(usa_pd)

# 6) Convert imperial -> metric units
usa_pd = convert_imperial_to_metric(usa_pd)

# 7) Save processed dataframe
usa_pd.to_csv(OUTPUT_PROCESSED, index=False)
print(f"Saved processed dataset to {OUTPUT_PROCESSED} (shape {usa_pd.shape})")

# 8) Quick peek
print(usa_pd.head(5))
