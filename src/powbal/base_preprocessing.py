import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


from sklearn.base import BaseEstimator, TransformerMixin

class HandleMissingReward(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["reward"] = X["reward"].fillna(0.0)
        return X

class HandleMissingPreSwitchReading(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "pre_switch_off_reading" in X.columns:
            median_val = X["pre_switch_off_reading"].median()
            X["pre_switch_off_reading"] = X["pre_switch_off_reading"].fillna(median_val)
        return X

class HandleMissingPowerNonzero(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "power_nonzero" in X.columns:
            X["power_nonzero"] = X["power_nonzero"].fillna(0.0)
        return X

class HandleMissingSurveyAnswer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "surveyanswer" in X.columns:
            X["surveyanswer"] = X["surveyanswer"].fillna("no response")
        return X



handle_missing_pipeline = Pipeline([
    ('reward_imputer', HandleMissingReward()),
    ('pre_switch_imputer', HandleMissingPreSwitchReading()),
    ('power_nonzero_imputer', HandleMissingPowerNonzero()),
    ('surveyanswer_imputer', HandleMissingSurveyAnswer())
])



def add_during_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary column 'is_during_event' indicating if the observation occurs during
    the predefined switch-off event window (between 08:00 and 23:59).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing an 'hour' column.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with new 'is_during_event' column.

    Example
    -------
    >>> df = add_during_event(df)
    """
    df = df.copy()
    event_cols = [f"switch_off_L{i}" for i in range(1, 7)]
    df["during_event"] = df[event_cols].any(axis=1)
    return df



def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical variables 'preference' and 'surveyanswer'.

    - Converts to lowercase.
    - Removes leading/trailing spaces.
    - Maps known inconsistent labels to standardized ones.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'preference' and/or 'surveyanswer' columns.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with standardized categorical values.

    Example
    -------
    >>> df = standardize_categoricals(df)
    """
    df = df.copy()
    if "preference" in df.columns:
        df["preference"] = df["preference"].astype(str).str.strip().str.lower()
        df["preference"] = df["preference"].replace({
            "turns on": "turns on",
            "remains off": "remains off"
        })
    if "surveyanswer" in df.columns:
        df["surveyanswer"] = df["surveyanswer"].astype(str).str.strip().str.lower()
    return df



# def conditional_event_based_imputation(df, cols_to_impute, event_cols):
#     """
#     Impute missing values in specific columns with 0.0
#     only when no switch-off event occurred in the row.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The input dataframe.
#     cols_to_impute : list of str
#         The columns to impute.
#     event_cols : list of str
#         The event indicator columns (switch_off_Lx and Fx).

#     Returns
#     -------
#     pd.DataFrame
#         The updated dataframe with conditional imputations.
#     """
#     df = df.copy()
#     df["any_switch_event"] = df[event_cols].sum(axis=1) > 0
#     for col in cols_to_impute:
#         mask = df[col].isna() & (~df["any_switch_event"])
#         df.loc[mask, col] = 0.0
#     df.drop(columns="any_switch_event", inplace=True)
#     return df

# def apply_conditional_event_imputation(df):
#     df = df.copy()
#     cols_to_impute = [
#         "off", "on", "notify_participant_at", "posted_to_api", "deactivated_at",
#         "switch_off", "switch_on", "notification", "notification_time", "switch_off_hour", "notification_hour",
#         "average_power_before_switchoff", "mean_W_before_switchoff",
#         "avoided_energy_consumption_Wh", "pre_switch_off_reading"
#     ]
#     event_cols = [f"switch_off_L{i}" for i in range(1, 7)] + [f"switch_off_F{i}" for i in range(1, 17)]
#     return conditional_event_based_imputation(df, cols_to_impute, event_cols)

def run_base_preprocessing(df):
    df = standardize_categoricals(df)
    df = handle_missing_pipeline.fit_transform(df)
    df = add_during_event(df)
    return df
