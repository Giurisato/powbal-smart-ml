import pandas as pd

def parse_and_extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'round_datetime' column and extract consistent datetime features.

    This function ensures that the time characteristics are derived consistently from the “round_datetime” column,
    avoiding duplications and ensuring consistency in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'round_datetime' column in string format.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime and extracted features: 'parsed_datetime', 'hour',
        'half_hour', 'month', 'dayofweek', and 'is_weekend'.

    Example
    -------
    df = parse_and_extract_datetime_features(df)
    """
    df['parsed_datetime'] = pd.to_datetime(df['round_datetime'], format="%d%b%Y %H:%M:%S", errors='coerce')
    df['hour'] = df['parsed_datetime'].dt.hour
    df['half_hour'] = df['parsed_datetime'].dt.hour * 2 + df['parsed_datetime'].dt.minute // 30 + 1
    df['month'] = df['parsed_datetime'].dt.month
    df['dayofweek'] = df['parsed_datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6])
    return df

def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'season' categorical feature based on the 'month'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'month' column (1–12).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'season' column.

    Example
    -------
    df = add_season(df)
    """
    def month_to_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        elif month in [3, 4, 5]:
            return "Summer-Pre monsoon"
        else:
            return "Autumn-Post_monsoon"

    df["season"] = df["month"].apply(month_to_season)
    return df

def add_part_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical 'part_of_day' feature based on the 'hour' column.

    Parameters
    df : pd.DataFrame
        Input dataframe containing an 'hour' column (int, 0–23).

    Returns
    pd.DataFrame
        DataFrame with new 'part_of_day' column added.

    Example
    df = add_part_of_day(df)
    """
    def map_hour_to_part(hour):
        if 5 <= hour <= 7:
            return "Early Morning"
        elif 8 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 16:
            return "Afternoon"
        elif 17 <= hour <= 20:
            return "Evening"
        elif 21 <= hour <= 23:
            return "Night"
        else:
            return "Late Night"

    df["part_of_day"] = df["hour"].apply(map_hour_to_part)
    return df

def add_is_peak_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary feature 'is_peak_hour' based on typical residential peak usage hours.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'hour' column (0–23).

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'is_peak_hour' boolean column.

    Example
    -------
    df = add_is_peak_hour(df)
    """
    df["is_peak_hour"] = df["hour"].isin([8, 9, 19, 20, 21])
    return df

def add_days_since_trial_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute number of days since each user's registration date.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'registered_at' and 'parsed_datetime' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'days_since_trial_start' column.

    Example
    -------
    df = add_days_since_trial_start(df)
    """
    df["registered_at"] = pd.to_datetime(df["registered_at"], errors='coerce')
    df["days_since_trial_start"] = (df["parsed_datetime"] - df["registered_at"]).dt.days
    return df

def add_rolling_daily_reward(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Add a 7-day rolling average of daily_reward per user.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame containing 'ca_number', 'parsed_datetime', and 'daily_reward'.

    window : int, optional
        Rolling window size in days (default is 7).

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with new column 'rolling_daily_reward'.

    Notes
    -----
    Assumes 'parsed_datetime' is a datetime column.
    """
    df = df.copy()
    if "parsed_datetime" in df.columns and "ca_number" in df.columns:
        df = df.sort_values(by=["ca_number", "parsed_datetime"])
        df["rolling_daily_reward"] = df.groupby("ca_number")["reward"].transform(
            lambda x: x.fillna(0).rolling(window=48, min_periods=1).mean()
        )
    return df

def clean_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant or duplicated temporal features to standardize time-related columns.

    This function drops columns that are either:
    - duplicates of already derived features from `parsed_datetime`
    - unused timestamp variants
    - irrelevant numeric encodings of time

    It retains essential features such as:
    - parsed_datetime and derived components (hour, month, dayofweek, is_weekend, half_hour)
    - event-specific timestamps (switch_off_time, switch_on_time)
    - the categorical 'day' (weekday name) for potential visualizations

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with mixed temporal columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with standardized temporal features.

    Example
    -------
    df = clean_temporal_columns(df)
    """
    cols_to_drop = [
        "timestamp_id", "datetime",
        "month", "monthofyear", "minute", "minute_all",
        "week", "weekofyear", "period_id"
    ]

    existing_cols = df.columns.intersection(cols_to_drop)
    df = df.drop(columns=existing_cols)
    return df

def run_datetime_parsing_and_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full datetime feature extraction and temporal column cleaning.

    This function:
    1. Parses 'round_datetime' into a datetime object
    2. Extracts hour, half_hour, month, dayofweek, is_weekend
    3. Adds season, is_peak_hour, days_since_trial_start
    4. Removes redundant and duplicate temporal columns

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw datetime-related columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with consistent datetime features and no redundant columns.

    Example
    -------
    df = run_datetime_parsing_and_cleaning(df)
    """
    df = parse_and_extract_datetime_features(df)
    df = add_season(df)
    df = add_part_of_day(df)
    df = add_is_peak_hour(df)
    df = add_days_since_trial_start(df)
    df = add_rolling_daily_reward(df)
    df = clean_temporal_columns(df)
    return df
