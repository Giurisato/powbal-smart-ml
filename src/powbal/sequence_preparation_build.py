import numpy as np
import pandas as pd


def build_sequences_for_modeling(
    df: pd.DataFrame,
    features: list,
    sequence_length: int = 48,
    groupby_col: str = "ca_number",
    datetime_col: str = "parsed_datetime",
    event_flag_col: str = None,
    mode: str = "general"
):
    """
    Builds time sequences for LSTM/CNN models, with the option to focus only on real events.

    Parametri
    ----------
    df : pd.DataFrame
        The time dataframe with one row for each timestep (already preprocessed).
    features : list
        List of numeric columns to be included as model input.
    sequence_length : int
        Length of the time window (e.g. 48 for a full day with 30-minute slots).
    groupby_col : str
        Column identifying the group on which to build the sequences (default: user 'ca_number').
    datetime_col : str
        Time column for sorting data (default: 'parsed_datetime').
    event_flag_col : str
        Boolean column indicating the presence of an event in the middle of the sequence (default: None).
        Used only if mode="event_only".
    mode : str
        "general": constructs all valid sequences (default).
        "event_only": builds only sequences with a real event at the centre, using event_flag_col.

    Ritorna
    -------
    X : np.ndarray
        Tensore delle sequenze con shape (n_seq, sequence_length, n_features).
    user_ids : list
        List of user IDs associated with each sequence.
    event_times : list or None
        Central timestamp of the sequence (only if mode="event_only").
    event_positions : list or None
        Index of the event's position within the group (only if mode="event_only").

    Note metodologiche
    ------------------
    The decision to use `switch_off_F1 == 1` as `event_flag_col` is motivated by:
    - is the closest flag in the past (30 minutes ago) and guarantees that the event has just occurred
    - corresponds to the central position in the window [t-24, ..., t+23] with central t = event
    - is supported by works such as Zhou et al. (2022), which construct sequences around the actual event

    Conversely, using generic `has_event` or `switch_off_Fx` does not guarantee that the event is in the centre position.
    and would therefore compromise the temporal consistency of the model. Per questo motivo, `switch_off_F1` è la scelta             consigliata.
    """

    df_sorted = df.sort_values([groupby_col, datetime_col])
    X = []
    user_ids = []
    event_times = []
    event_positions = []

    for uid, group in df_sorted.groupby(groupby_col):
        group = group.reset_index(drop=True)
        data = group[features].values

        if mode == "general":
            if len(data) >= sequence_length:
                for i in range(len(data) - sequence_length + 1):
                    X.append(data[i:i + sequence_length])
                    user_ids.append(uid)

        elif mode == "event_only":
            if event_flag_col is None or event_flag_col not in group.columns:
                raise ValueError("Per mode='event_only' devi fornire event_flag_col valida")

            valid_indices = group.index[group[event_flag_col] == 1].tolist()

            for i in valid_indices:
                start = i - sequence_length // 2
                end = i + sequence_length // 2
                if start >= 0 and end <= len(group):
                    window = data[start:end]
                    X.append(window)
                    user_ids.append(uid)
                    event_times.append(group.loc[i, datetime_col])
                    event_positions.append(i)

    X = np.array(X)
    return X, user_ids, event_times if mode == "event_only" else None, event_positions if mode == "event_only" else None
