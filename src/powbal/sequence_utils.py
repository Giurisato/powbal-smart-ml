

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Adapt import to the layout (es. "from powbal.sequence_preparation ...")
from sequence_preparation import (
    build_sequences_for_modeling,
    extract_event_timing_features,
)


TIMING_FEATURES = {
    "switch_off_hour",
    "switch_on_hour",
    "notification_hour",
    "notification_to_switch_off_mins",
    "switch_off_to_on_mins",
}


def _ensure_timing_features_if_needed(df, requested_features: List[str], verbose: bool = True):
    """
    If the user has added columns to the features that originate from
    extract_event_timing_features and it is not present in the df, I will calculate it now.
    """
    needs_timing = any((c in TIMING_FEATURES) and (c not in df.columns) for c in requested_features)
    if needs_timing:
        if verbose:
            print("  Mancano timing-features richieste: eseguo extract_event_timing_features() ...")
        df = extract_event_timing_features(df)  # aggiunge le colonne mancanti se i time-col sono disponibili
    return df


def _intersect_features(df, features: List[str], verbose: bool = True) -> List[str]:
    """
    Cross-reference the required features with the available columns, highlighting any missing ones.
    """
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if verbose and missing:
        print(f"  Le seguenti features richieste non sono nel DataFrame e verranno ignorate: {missing}")
    if not present:
        raise ValueError("Nessuna delle features richieste è presente nel DataFrame dopo i controlli.")
    return present


def build_sequences_chunked_fixed(
    df,
    features: List[str],
    fx_cols: List[str],
    save_dir: str,
    sequence_length: int = 48,
    override_col: str = "override",
    user_col: str = "ca_number",
    datetime_col: str = "parsed_datetime",
    save_meta: bool = True,
    save_npz_compressed: bool = False,
    ensure_event_center: bool = False,  # utile solo per F1 se vuoi un check rigoroso
    verbose: bool = True,
) -> None:
    """
    Generate event-centric chunks (one for each Fx) and save:
    - X_chunk_k.npy: float32, shape (n_seq, T, F)
    - y_chunk_k.npy: int8,    shape (n_seq,) con 0/1 (override al centro)
    - meta_chunk_k.npz (opzionale): features, fx_col, user_ids/event_times filtrati

    y is defined as X[:, centre_idx, idx_override], binarised (>0.5 → 1).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Timing features on-demand (se richieste ma assenti)
    df = _ensure_timing_features_if_needed(df, features, verbose=verbose)

    # Features effettive presenti
    features = _intersect_features(df, features, verbose=verbose)

    if override_col not in features:
        raise ValueError(f"`override_col='{override_col}' non è in features: {features}")

    center_idx = sequence_length // 2
    idx_override = features.index(override_col)

    chunk_id = 0
    for fx in fx_cols:
        if verbose:
            print(f"\n Costruzione sequenze per: {fx}")

        # Costruzione sequenze centrata sull'indice in cui fx==1
        X_tmp, user_ids, event_times, event_positions = build_sequences_for_modeling(
            df=df,
            features=features,
            sequence_length=sequence_length,
            groupby_col=user_col,
            datetime_col=datetime_col,
            event_flag_col=fx,
            mode="event_only",
        )

        if X_tmp is None or getattr(X_tmp, "size", 0) == 0:
            if verbose:
                print(f"  Nessuna sequenza trovata per {fx}, skip.")
            continue

        X_tmp = np.asarray(X_tmp)
        if X_tmp.ndim != 3 or X_tmp.shape[1] != sequence_length:
            raise ValueError(f"Shape inattesa per X_tmp: {X_tmp.shape}, atteso (n_seq, {sequence_length}, n_features).")

        # y = override al centro
        y_tmp = X_tmp[:, center_idx, idx_override]

        # (Opz.) check che l'evento sia 1 al centro — utile per F1, non per F2..F16.
        if ensure_event_center and fx in features:
            idx_fx = features.index(fx)
            center_flag = X_tmp[:, center_idx, idx_fx]
            if np.nanmax(center_flag) <= 0:
                print(f"  Warning: per {fx} il flag evento non risulta 1 al centro. (Atteso per F1, non per F2..F16)")

        # Filtra sequenze con y NaN (override mancante al centro)
        valid_mask = ~np.isnan(y_tmp)
        X_tmp = X_tmp[valid_mask].astype(np.float32)
        y_tmp = (y_tmp[valid_mask] > 0.5).astype(np.int8)

        # Salvataggi
        x_path = os.path.join(save_dir, f"X_chunk_{chunk_id}.npy")
        y_path = os.path.join(save_dir, f"y_chunk_{chunk_id}.npy")
        np.save(x_path, X_tmp)
        np.save(y_path, y_tmp)

        if save_meta:
            meta_out: Dict[str, Any] = {
                "fx_col": fx,
                "center_idx": center_idx,
                "features": np.array(features),
            }
            # Filtra user_ids/event_times se disponibili e allineati
            if user_ids is not None and len(user_ids) == len(valid_mask):
                meta_out["user_ids"] = np.asarray(user_ids)[valid_mask]
            if event_times is not None and len(event_times) == len(valid_mask):
                meta_out["event_times"] = np.asarray(event_times)[valid_mask]

            np.savez(os.path.join(save_dir, f"meta_chunk_{chunk_id}.npz"), **meta_out)

        if save_npz_compressed:
            np.savez_compressed(os.path.join(save_dir, f"chunk_{chunk_id}.npz"), X=X_tmp, y=y_tmp)

        # Log bilanciamento chunk
        vals, cnts = np.unique(y_tmp, return_counts=True)
        if verbose:
            dist = dict(zip(vals.tolist(), cnts.tolist()))
            print(f"✅ Salvato chunk {chunk_id} → X{X_tmp.shape}, y{y_tmp.shape} | Distribuzione override: {dist}")

        chunk_id += 1


def analyze_class_balance_from_chunks(save_dir: str) -> Tuple[int, int, float]:
    """
    Load all y_chunk_*.npy files and print the global balance.
    Return: (N_tot, N_pos, Pos_rate)
    """
    y_paths = sorted(glob.glob(os.path.join(save_dir, "y_chunk_*.npy")))
    if not y_paths:
        print("  Nessun y_chunk_*.npy trovato.")
        return 0, 0, 0.0

    ys = []
    for yp in y_paths:
        y = np.load(yp)
        if y.ndim != 1:
            y = y.reshape(-1)
        ys.append(y.astype(np.int8))

    y_final = np.concatenate(ys, axis=0)
    n_tot = int(y_final.size)
    n_pos = int(y_final.sum())
    pos_rate = n_pos / max(n_tot, 1)
    print(f" Bilanciamento globale: N={n_tot} | Pos={n_pos} ({pos_rate:.2%})")
    return n_tot, n_pos, pos_rate


def analyze_class_balance_per_chunk(save_dir: str) -> List[Dict[str, Any]]:
    """
    Returns a list with balancing for each chunk, including (if available) the fx column.
    """
    results = []
    y_paths = sorted(glob.glob(os.path.join(save_dir, "y_chunk_*.npy")))
    for yp in y_paths:
        y = np.load(yp).astype(np.int8).reshape(-1)
        chunk_id = int(os.path.basename(yp).split("_")[-1].split(".")[0])
        meta_path = os.path.join(save_dir, f"meta_chunk_{chunk_id}.npz")
        fx_col = None
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            fx_col = str(meta.get("fx_col", None))
        res = {
            "chunk_id": chunk_id,
            "fx_col": fx_col,
            "n": int(y.size),
            "pos": int(y.sum()),
            "pos_rate": float(y.mean()),
        }
        results.append(res)
    # Log ordinato per pos_rate decrescente
    results.sort(key=lambda d: d["pos_rate"], reverse=True)
    for r in results:
        print(f"Chunk {r['chunk_id']:>2} | fx={r['fx_col']:<16} | N={r['n']:<6} | Pos={r['pos']:<6} | {r['pos_rate']:.2%}")
    return results


def load_all_chunks(save_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate all the X_chunk_*.npy e y_chunk_*.npy in X_final, y_final.
    """
    x_paths = sorted(glob.glob(os.path.join(save_dir, "X_chunk_*.npy")))
    y_paths = sorted(glob.glob(os.path.join(save_dir, "y_chunk_*.npy")))
    if len(x_paths) != len(y_paths):
        raise RuntimeError(f"Mismatch numero chunk: {len(x_paths)} X vs {len(y_paths)} y.")

    X_list, Y_list = [], []
    for xp, yp in zip(x_paths, y_paths):
        X = np.load(xp).astype(np.float32)
        y = np.load(yp).astype(np.int8).reshape(-1)
        X_list.append(X)
        Y_list.append(y)

    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(Y_list, axis=0)
    print(f" Merge: X_final{X_final.shape}, y_final{y_final.shape}")
    return X_final, y_final
