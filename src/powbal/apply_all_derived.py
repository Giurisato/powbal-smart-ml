from powbal.derived_features import (
    add_event_intensity,
    add_event_recent_history,
    add_active_today_flag,
    add_event_upcoming_3h,
    add_event_next_run_features,
    add_event_recent_count_8h,
    add_override_rolling_8h
)

def apply_all_derived_features(df):
    """
    Apply all core derived features transformations to the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing relevant switch-off and override features.

    Returns
    -------
    pd.DataFrame
        DataFrame with new derived features added.

    Example
    -------
    df = apply_all_derived_features(df)
    """
    df = add_event_intensity(df)
    df = add_event_recent_history(df)
    df = add_active_today_flag(df)
    df = add_event_upcoming_3h(df)
    df = add_event_next_run_features(df)
    df = add_event_recent_count_8h(df)
    df = add_override_rolling_8h(df)
    return df
