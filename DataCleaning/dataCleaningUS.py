import warnings
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)

# ---------- Configuration ----------
# Use relative paths to Data folder
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
INPUT_CSV = os.path.join(DATA_DIR, "US_Accidents_March23_sampled_500k.csv")
OUTPUT_PROCESSED = os.path.join(DATA_DIR, "US_Accidents_processed_for_modeling.csv")
OUTPUT_MODEL_SUBSET = os.path.join(DATA_DIR, "US_Accidents_model_subset.csv")
DROP_MISSING_COL_THRESHOLD = 0.40  # drop columns with >40% missing
# columns that are noisy / textual and safe to drop before processing
TEXT_DROP_CANDIDATES = ['ID', 'Description', 'Street', 'Weather_Timestamp', 'Source', 'Country']
# columns known near-constant/low-value to drop early
LOW_VALUE_CATEGORICAL = ['Astronomical_Twilight', 'Nautical_Twilight', 'Civil_Twilight']
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

def drop_low_value_columns(df, candidates):
    drop_cols = [c for c in candidates if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped low-value categorical columns: {drop_cols}")
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

def prune_geo_and_cardinality(df):
    """Remove problematic high-missing or redundant geo columns, keep compact geo signal.
    - Drop End_Lat/End_Lng (often ~44% missing)
    - Keep Start_Lat/Start_Lng and Distance(km) if present
    - Drop Zipcode, Airport_Code to reduce cardinality
    """
    cols_to_drop = [c for c in ['End_Lat', 'End_Lng', 'Zipcode', 'Airport_Code'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Pruned high-missing/high-cardinality geo columns: {cols_to_drop}")
    return df

def impute_weather_simple(df):
    """Simple imputation for common weather fields: median for numeric, mode for categorical.
    Only applied to a handful of columns to avoid heavy distortion.
    """
    num_cols = [c for c in ['Precipitation(mm)', 'Wind_Speed(m/s)', 'Visibility(km)', 'Temperature(C)', 'Pressure(hPa)'] if c in df.columns]
    for c in num_cols:
        med = pd.to_numeric(df[c], errors='coerce').median()
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(med)
    cat_cols = [c for c in ['Weather_Condition', 'Wind_Direction'] if c in df.columns]
    for c in cat_cols:
        mode = df[c].mode(dropna=True)
        if not mode.empty:
            df[c] = df[c].fillna(mode.iloc[0])
    print("Applied simple weather imputation (median/mode) for selected columns.")
    return df

def encode_time_cyclical(df):
    """Encode hour/day/month as cyclical features if start_* exist."""
    if 'start_hour' in df.columns:
        df['start_hour_sin'] = np.sin(2 * np.pi * df['start_hour'] / 24)
        df['start_hour_cos'] = np.cos(2 * np.pi * df['start_hour'] / 24)
    if 'start_dayofweek' in df.columns:
        df['start_dow_sin'] = np.sin(2 * np.pi * df['start_dayofweek'] / 7)
        df['start_dow_cos'] = np.cos(2 * np.pi * df['start_dayofweek'] / 7)
    if 'start_month' in df.columns:
        df['start_month_sin'] = np.sin(2 * np.pi * df['start_month'] / 12)
        df['start_month_cos'] = np.cos(2 * np.pi * df['start_month'] / 12)
    return df

def add_zero_flags_and_log(df):
    """For skewed numerics, add zero flags and log1p transforms if present."""
    skew_cols_map = {
        'Precipitation(mm)': 'precip_zero',
        'Distance(km)': 'distance_zero',
        'Wind_Speed(m/s)': 'windspeed_zero'
    }
    for col, flag in skew_cols_map.items():
        if col in df.columns:
            df[flag] = (pd.to_numeric(df[col], errors='coerce') == 0).astype(int)
            df[col + '_log1p'] = np.log1p(pd.to_numeric(df[col], errors='coerce'))
    return df

def drop_constant_columns(df):
    """Drop columns that are constant in this sample."""
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols)
        print(f"Dropped constant columns: {const_cols}")
    return df

def drop_duplicates(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if after != before:
        print(f"Dropped duplicates: {before - after}")
    return df

# ---------- Pipeline (simple linear sequence) ----------
# 1) Drop noisy text/id columns
usa_pd = drop_text_columns(usa_pd, TEXT_DROP_CANDIDATES)
usa_pd = drop_low_value_columns(usa_pd, LOW_VALUE_CATEGORICAL)

# 2) Parse datetime-like columns (safe fallback parsing implemented)
usa_pd = parse_datetimes(usa_pd)

# 3) Drop high-missing columns
usa_pd, dropped_cols = drop_sparse_columns(usa_pd, DROP_MISSING_COL_THRESHOLD)
usa_pd = prune_geo_and_cardinality(usa_pd)

# 4) Lightweight imputation for key weather fields, then drop remaining NA rows
usa_pd = impute_weather_simple(usa_pd)
usa_pd = drop_rows_with_any_na(usa_pd)

# 5) Feature engineering: simple time features and duration
usa_pd = add_time_features(usa_pd)
usa_pd = encode_time_cyclical(usa_pd)
usa_pd = add_zero_flags_and_log(usa_pd)

# 6) Convert imperial -> metric units
usa_pd = convert_imperial_to_metric(usa_pd)
usa_pd = drop_constant_columns(usa_pd)
usa_pd = drop_duplicates(usa_pd)

# 7) Save processed dataframe and a stratified modeling subset if Severity exists
usa_pd.to_csv(OUTPUT_PROCESSED, index=False)
print(f"Saved processed dataset to {OUTPUT_PROCESSED} (shape {usa_pd.shape})")

if 'Severity' in usa_pd.columns:
    # create a manageable stratified subset for prototyping
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
        X = usa_pd
        y = usa_pd['Severity']
        for _, idx in sss.split(X, y):
            subset = usa_pd.iloc[idx].copy()
        subset.to_csv(OUTPUT_MODEL_SUBSET, index=False)
        print(f"Saved stratified model subset to {OUTPUT_MODEL_SUBSET} (shape {subset.shape})")
    except Exception as e:
        warnings.warn(f"Failed to create stratified subset: {e}")

# 8) Quick peek
print(usa_pd.head(5))

# ---------- Alternative pipeline for clustering (no data cleaning, no precipitation modification) ----------
def clean_for_clustering(input_path, output_path):
    """Load raw data and save without modifying precipitation.
    Skips all data cleaning steps - useful for clustering on raw accident data.
    """
    df = pd.read_csv(input_path)
    print(f"\n[CLUSTERING PIPELINE] Loaded dataset: rows={df.shape[0]:,d}, cols={df.shape[1]}")
    print(f"[CLUSTERING PIPELINE] No data cleaning applied - saving raw data with original precipitation units")
    
    # Save as-is for clustering
    df.to_csv(output_path, index=False)
    print(f"[CLUSTERING PIPELINE] Saved raw dataset to {output_path} (shape {df.shape})")
    
    return df

# Run clustering pipeline if desired (commented out; uncomment to generate)
clustering_output = os.path.join(DATA_DIR, "US_Accidents_for_clustering.csv")
clean_for_clustering(INPUT_CSV, clustering_output)
