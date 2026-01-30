import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from powbal.lstm_attention_model import LSTMAttentionNet
from powbal.dataset_lstm import EventSequenceDataset

import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score


def train_lstm_model(X, y, mask=None, user_ids=None,
                     input_dim=8, hidden_dim=64, batch_size=64,
                     lr=1e-3, epochs=20, save_path="model_lstm.pt"):
    """
    Train the LSTM + Attention model on event-centric sequences.

    Parametri
    ----------
    X : np.ndarray
        Sequenze di input (n_seq, seq_len, n_feat)
    y : np.ndarray
        Target (n_seq,)
    mask : np.ndarray, opzionale
        Maschera attention (n_seq, seq_len)
    user_ids : list, opzionale
        Lista utenti per tracking/debug

    Output
    ------
    model : modello PyTorch addestrato
    history : lista delle loss per epoca
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset + loader
    dataset = EventSequenceDataset(X, y, mask=mask, user_ids=user_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modello
    model = LSTMAttentionNet(input_dim=input_dim, hidden_dim=hidden_dim)
    model.to(device)

    # Ottimizzazione
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            x = batch['x'].to(device)
            y_true = batch['y'].to(device)

            optimizer.zero_grad()
            y_pred, attn_weights = model(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            all_preds.append(torch.sigmoid(y_pred).detach().cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

        epoch_loss = running_loss / len(dataset)
        history.append(epoch_loss)

        preds_bin = (np.concatenate(all_preds) > 0.5).astype(int)
        targets_bin = np.concatenate(all_targets).astype(int)
        acc = accuracy_score(targets_bin, preds_bin)
        auc = roc_auc_score(targets_bin, np.concatenate(all_preds))

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {acc:.4f} - AUC: {auc:.4f}")

    torch.save(model.state_dict(), save_path)
    return model, history
