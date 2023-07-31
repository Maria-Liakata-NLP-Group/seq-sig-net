from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

class BiLSTM_stacked(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        hidden_dim_lstm1: int,
        hidden_dim_lstm2: int, 
        output_dim: int, 
        dropout_rate: float,
        pad_with: int = 0):

        """
        Parameters
        ----------
        embedding_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        hidden_dim_lstm1: int
            Dimensions of the hidden layer in the first BiLSTM.
        hidden_dim_lstm2: int
            Dimensions of the hidden layer in the second BiLSTM.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.
        pad_with : int, optional
            integer with which we are padding the windows. default: 0
        """

        super(BiLSTM_stacked, self).__init__()

        self.pad_with = pad_with
        self.hidden_dim_lstm1 = hidden_dim_lstm1
        self.hidden_dim_lstm2 = hidden_dim_lstm2
        #BiLSTMs
        self.lstm1 = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim_lstm1, num_layers=1, batch_first=True, bidirectional=True).double()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_lstm1, hidden_size=hidden_dim_lstm2, num_layers=1, batch_first=True, bidirectional=True).double()
        self.fc3 = nn.Linear(hidden_dim_lstm2, output_dim)
    
    def forward(self, x):
        seq_lengths = torch.sum(x[:, :, 0] != self.pad_with, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]

        #BiLSTM 1
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
        out, (_, _) = self.lstm1(x_pack)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        inverse_perm = np.argsort(perm_idx)
        out = out[inverse_perm]
        out = out[:, :, :self.hidden_dim_lstm1] + out[:, :, self.hidden_dim_lstm1:]

        out = self.dropout(out)

        #BiLSTM 2
        out = out[perm_idx]
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(out, seq_lengths, batch_first=True)
        outl, (out_h, _) = self.lstm2(x_pack)
        outl, _ = torch.nn.utils.rnn.pad_packed_sequence(outl, batch_first=True)
        outl = outl[inverse_perm]
        out = out_h[-1, :, :] + out_h[-2, :, :]
        out = out[inverse_perm]

        out = self.dropout(out)

        out = self.fc3(out.float())
        return out