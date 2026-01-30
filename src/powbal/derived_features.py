# derived_features.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import pandas as pd



def add_event_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'event_intensity' as the total number of upcoming switch-off flags
    in the next 3 hours (6 half-hour slots): sum(switch_off_L1..L6).

    Returns
    -------
    pd.DataFrame with a new column:
        - event_intensity : int in [0..6]
    """
    lead_cols = [f"switch_off_L{i}" for i in range(1, 7)]
    df["event_intensity"] = df[lead_cols].sum(axis=1).astype(int)
    return df


def add_event_recent_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'event_recent_history' as the total number of switch-off flags
    observed in the past 8 hours (16 half-hour slots): sum(switch_off_F1..F16).

    Returns
    -------
    pd.DataFrame with a new column:
        - event_recent_history : int in [0..16]
    """
    lag_cols = [f"switch_off_F{i}" for i in range(1, 17)]
    df["event_recent_history"] = df[lag_cols].sum(axis=1).astype(int)
    return df


def add_active_today_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience flag for visual summaries: 'active_today' is True if the slot
    accrued any reward (>0). This is a lightweight per-slot indicator, not a
    daily aggregate.

    Returns
    -------
    pd.DataFrame with:
        - active_today : bool
    """
    df["active_today"] = (df["reward"].fillna(0) > 0)
    return df



def add_event_upcoming_3h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clean 3-hour upcoming-event features based on lead flags L1..L6.

    Columns created
    ---------------
    - event_upcoming_count_3h : int in [0..6], sum of switch_off_L1..L6
    - event_upcoming_cat_3h   : categorical in {'none','one','two_plus'}

    Rationale
    ---------
    This replaces generic 'switchoff_length' naming with a clearer meaning:
    'how many events fall within the next 3 hours', independent of contiguity.
    """
    lead_cols = [f"switch_off_L{i}" for i in range(1, 7)]
    cnt = df[lead_cols].sum(axis=1).astype(int)
    df["event_upcoming_count_3h"] = cnt
    df["event_upcoming_cat_3h"] = pd.Categorical(
        np.where(cnt == 0, "none", np.where(cnt == 1, "one", "two_plus")),
        categories=["none", "one", "two_plus"],
        ordered=True
    )
    return df


def add_event_next_run_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features describing the *continuous* run length of the very next
    upcoming switch-off block, starting at L1.

    Columns created
    ---------------
    - event_next_run_length : int in {0,1,2,3,4,5,6}
        Number of consecutive ones from [L1,L2,...] until the first zero.
        Example: [1,1,0,1,0,0] -> 2
    - event_next_run_class  : categorical in {'none','1_slot','2+_slots'}

    Notes
    -----
    This complements 'event_upcoming_count_3h' by focusing on contiguity
    (duration of the first imminent event block).
    """
    lead_cols = [f"switch_off_L{i}" for i in range(1, 7)]

    def _runlen_first_block(row: pd.Series) -> int:
        vals = row.values.astype(int)
        length = 0
        for v in vals:
            if v == 1:
                length += 1
            else:
                break
        return length

    runlen = df[lead_cols].apply(_runlen_first_block, axis=1).astype(int)
    df["event_next_run_length"] = runlen
    df["event_next_run_class"] = pd.Categorical(
        np.where(runlen == 0, "none", np.where(runlen == 1, "1_slot", "2+_slots")),
        categories=["none", "1_slot", "2+_slots"],
        ordered=True
    )
    return df


# ---------------------------------------------------------------------
# Explicit “recent” alias @ 8h for clarity in dashboards
# ---------------------------------------------------------------------
def add_event_recent_count_8h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'event_recent_count_8h' as an explicit alias (recomputed) of the
    8-hour history: sum(switch_off_F1..F16). Kept separate from
    'event_recent_history' for clearer dashboard labelling.

    Returns
    -------
    pd.DataFrame with:
        - event_recent_count_8h : int in [0..16]
    """
    lag_cols = [f"switch_off_F{i}" for i in range(1, 17)]
    df["event_recent_count_8h"] = df[lag_cols].sum(axis=1).astype(int)
    return df


# ---------------------------------------------------------------------
# Rolling override count over 8 hours
# ---------------------------------------------------------------------
def add_override_rolling_8h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user-level rolling 8-hour sum of 'override' actions
    (16 half-hour slots), sorted by user and timestamp.

    Columns required
    ----------------
    - ca_number
    - parsed_datetime  (parsable to datetime)
    - override         (0/1)

    Column created
    --------------
    - override_rolling_8h : float (rolling sum, min_periods=1)

    Notes
    -----
    The 8h window aligns with the trial's most relevant notice horizon and
    keeps interpretation consistent across profiling charts.
    """
    df = df.copy()
    df["parsed_datetime"] = pd.to_datetime(df["parsed_datetime"], errors="coerce")
    df = df.sort_values(["ca_number", "parsed_datetime"])

    df["override_rolling_8h"] = (
        df.groupby("ca_number", group_keys=False)["override"]
          .rolling(window=16, min_periods=1)
          .sum()
          .reset_index(level=0, drop=True)
    )
    return df
