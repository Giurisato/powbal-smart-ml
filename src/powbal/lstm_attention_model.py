import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Simple attention mechanism for temporal interpretability.
    Reference: Luong et al., 2015
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        energy = self.attn(lstm_output)  # (batch_size, seq_len, 1)
        weights = torch.softmax(energy.squeeze(-1), dim=1)  # (batch_size, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim)
        return context, weights


class LSTMAttentionNet(nn.Module):
    """
    LSTM + Attention model for sequence classification or regression.

    Usage:
    - Input: (batch_size, seq_len, input_dim)
    - Output: prediction + attention weights

    Based on:
    - Zhou et al., 2022 (event-centered sequences)
    - Luong et al., 2015 (temporal attention)
    - pytorch-attention-mechanism (GitHub)
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        context, attn_weights = self.attention(lstm_out)  # (batch_size, hidden_dim), (batch_size, seq_len)
        output = self.fc(context)  # (batch_size, output_dim)
        return output.squeeze(-1), attn_weights  # (batch,), (batch, seq_len)
