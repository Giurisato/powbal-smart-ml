import numpy as np
import pandas as pd


def extract_event_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae feature temporali da colonne evento (switch_on/off, notification).
    Aggiunge:
    - switch_off_hour, switch_on_hour, notification_hour
    - notification_to_switch_off_mins
    - switch_off_to_on_mins
    """
    df = df.copy()
    time_cols = ['switch_off_time', 'switch_on_time', 'notification_time']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d%b%Y %H:%M:%S", errors="coerce")

    df["switch_off_hour"] = df["switch_off_time"].dt.hour
    df["switch_on_hour"] = df["switch_on_time"].dt.hour
    df["notification_hour"] = df["notification_time"].dt.hour

    df["notification_to_switch_off_mins"] = (
        (df["switch_off_time"] - df["notification_time"]).dt.total_seconds() / 60
    )
    df["switch_off_to_on_mins"] = (
        (df["switch_on_time"] - df["switch_off_time"]).dt.total_seconds() / 60
    )

    return df

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
    Costruisce sequenze temporali per modelli LSTM/CNN, con opzione per focalizzarsi solo sugli eventi reali.

    Parametri
    ----------
    df : pd.DataFrame
        The temporal dataframe with one raw for every timestep (già preprocessato).
    features : list
        Lista delle colonne numeriche da includere come input del modello.
    sequence_length : int
        Lunghezza della finestra temporale (es. 48 per una giornata intera a slot di 30 min).
    groupby_col : str
        Colonna che identifica il gruppo su cui costruire le sequenze (default: utente 'ca_number').
    datetime_col : str
        Colonna temporale per ordinare i dati (default: 'parsed_datetime').
    event_flag_col : str
        Colonna booleana che indica la presenza di un evento al centro della sequenza (default: None).
        Usata solo se mode="event_only".
    mode : str
        "general": costruisce tutte le sequenze valide (default).
        "event_only": costruisce solo le sequenze con evento reale al centro, usando event_flag_col.

    Ritorna
    -------
    X : np.ndarray
        Tensore delle sequenze con shape (n_seq, sequence_length, n_features).
    user_ids : list
        Lista degli ID utente associati a ciascuna sequenza.
    event_times : list or None
        Timestamp centrale della sequenza (solo se mode="event_only").
    event_positions : list or None
        Indice della posizione dell'evento all'interno del gruppo (solo se mode="event_only").

    Note metodologiche
    ------------------
    La scelta di usare `switch_off_F1 == 1` come `event_flag_col` è motivata da:
    - è il flag più vicino nel passato (30 min fa) e garantisce che l'evento sia appena avvenuto
    - corrisponde alla posizione centrale nella finestra [t-24, ..., t+23] con t centrale = evento
    - è supportato da lavori come Zhou et al. (2022), che costruiscono sequenze attorno all'evento reale

    Al contrario, usare `has_event` o `switch_off_Fx` generici non garantisce che l'evento sia nella posizione centrale
    e quindi comprometterebbe la coerenza temporale del modello. Per questo motivo, `switch_off_F1` è la scelta consigliata.
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




# def build_sequences_for_modeling(
#     df: pd.DataFrame,
#     features: list,
#     sequence_length: int = 48,
#     groupby_col: str = "ca_number",
#     datetime_col: str = "parsed_datetime",
#     mode: str = "general",
#     event_flag_col: str = "switch_off_F1"
# ):
#     """
#     Costruisce sequenze temporali standard (CNN/LSTM full).
#     Se mode="event", filtra le righe dove event_flag_col == 1.
#     """
#     df_sorted = df.sort_values([groupby_col, datetime_col])
#     if mode == "event":
#         df_sorted = df_sorted[df_sorted[event_flag_col] == 1]

#     X = []
#     user_ids = []

#     for uid, group in df_sorted.groupby(groupby_col):
#         group = group.reset_index(drop=True)
#         data = group[features].values

#         if len(data) >= sequence_length:
#             for i in range(0, len(data) - sequence_length + 1):
#                 window = data[i:i + sequence_length]
#                 X.append(window)
#                 user_ids.append(uid)

#     X = np.array(X)
#     return X, user_ids


# def build_event_centered_sequences(
#     df: pd.DataFrame,
#     features: list,
#     sequence_length: int = 48,
#     event_flag_col: str = "switch_off_F1",
#     groupby_col: str = "ca_number",
#     datetime_col: str = "parsed_datetime",
# ):
#     """
#     Costruisce sequenze temporali centrate sull'evento (event-centric windowing).
#     Ideale per LSTM con Attention su override o risposta a switch-off.
#     """
#     assert sequence_length % 2 == 0, "sequence_length deve essere pari"
#     half_window = sequence_length // 2

#     X = []
#     user_ids = []
#     event_times = []
#     event_positions = []

#     df_sorted = df.sort_values([groupby_col, datetime_col]).reset_index(drop=True)

#     for uid, group in df_sorted.groupby(groupby_col):
#         group = group.reset_index(drop=True)
#         event_indices = group.index[group[event_flag_col] == 1].tolist()

#         for idx in event_indices:
#             start = idx - half_window
#             end = idx + half_window
#             if start >= 0 and end < len(group):
#                 seq = group.iloc[start:end][features].values
#                 X.append(seq)
#                 user_ids.append(uid)
#                 event_times.append(group.loc[idx, datetime_col])
#                 event_positions.append(half_window)

#     X = np.array(X)
#     return X, user_ids, event_times, event_positions
