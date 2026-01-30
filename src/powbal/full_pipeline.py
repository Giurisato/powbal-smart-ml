import pandas as pd
from powbal.base_preprocessing import run_base_preprocessing
from powbal.datetime_features import run_datetime_parsing_and_cleaning
from powbal.apply_all_derived import apply_all_derived_features

def apply_all_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing and feature engineering pipeline to POWBAL smart switch dataset.

    Steps:
    1. Basic preprocessing (categorical standardization, missing value imputation, event-conditioned imputations, etc.)
    2. Datetime parsing and temporal feature cleanup (includes rolling_daily_reward)
    3. Derived feature creation (event intensity, override rolling count, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Fully transformed dataframe with cleaned and engineered features.

    Example
    -------
    >>> df_clean = apply_all_transformations(df_raw)
    """
    df = run_base_preprocessing(df)
    df = run_datetime_parsing_and_cleaning(df)  # includes add_rolling_daily_reward
    df = apply_all_derived_features(df)
    return df
