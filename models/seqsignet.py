from __future__ import annotations
import torch
import torch.nn as nn
import signatory
import numpy as np


class SeqSigNet(nn.Module):
    def __init__(
        self, 
        input_channels: int, 
        output_channels: int, 
        sig_d: int, 
        hidden_dim_lstm: list[int], 
        embedding_dim: int,
        hidden_dim: int, 
        output_dim: int, 
        dropout_rate: float, 
        augmentation_tp: str =  'Conv1d', 
        augmentation_layers: list[int] | int | None = (), 
        comb_method: str ='concatenation', 
        attention: bool =False,
        num_time_features: int = 1):

        """
        Parameters
        ----------
        input_channels : int
            Dimension of the (dimensonally reduced) history embeddings that will be passed in. 
        output_channels : int
            Requested dimension of the embeddings after convolution layer.
        sig_d : int
            The depth to truncate the path logsignature at.
        hidden_dim_lstm :  list[int]
            A list that contains: 1. Dimensions of the hidden layer for the LSTM in the SWNU blocks and 2. Dimensions of the hidden layers in the final BiLSTM applied to the output
            of the SWNU units.
        embedding_dim: int
            Dimensions of current BERT post embedding. Usually 384 or 768.
        hidden_dim: int
            Dimensions of the hidden layer in the FFN.
        output_dim : int
            Dimension of the output layer in the FFN.
        dropout_rate : float
            Dropout rate in the FFN.
        augmentation_tp : str, optional
            Method of augmenting the path, by default "Conv1d".
            Options are:
            - "Conv1d": passes path through 1D convolution layer.
            - "CNN": passes path through `Augment` layer from `signatory` package.
        augmentation_layers : list[int] | int | None
            Dimensions of the hidden layers in the augmentation layer.
            Passed into `Augment` class from `signatory` package if
            `augmentation_type='signatory'`, by default None.
        comb_method : str, optional
            Determines how to combine the path signature and embeddings,
            by default "gated_addition".
            Options are:
            - concatenation: concatenation of path signature and embedding vector
            - gated_addition: element-wise addition of path signature and embedding vector
            - gated_concatenation: concatenation of linearly gated path signature and embedding vector
            - scaled_concatenation: concatenation of single value scaled path signature and embedding vector
        attention: bool, optional
            Option for applying multihead self-attention with a default of 3 heads directly to the input (before the SWNU unit).
        num_time_features : int, optional
            Number of time features to add to FFN input. 
        """

        super(SeqSigNet, self).__init__()
        self.input_channels = input_channels
        self.attention = attention
        self.embedding_dim = embedding_dim
        self.num_time_features = num_time_features

        if comb_method not in ["concatenation", "gated_addition", "gated_concatenation", "scaled_concatenation"]:
            raise ValueError(
                "`comb_method` must be either 'concatenation' or 'gated_addition' "
                "or 'gated_concatenation' or 'scaled_concatenation'."
            )
        self.comb_method = comb_method

        #self attention
        self.self_attn = nn.MultiheadAttention(input_channels, num_heads=3, bias=True).double()

        #Convolution/augmentation
        if augmentation_tp not in ["Conv1d", "CNN"]:
            raise ValueError("`augmentation_tp` must be 'Conv1d' or 'signatory'.")
        self.augmentation_tp = augmentation_tp 
        self.conv = nn.Conv1d(input_channels, output_channels, 3, stride=1, padding=1).double()
        self.augment = signatory.Augment(in_channels=input_channels,
                    layer_sizes = augmentation_layers,
                    kernel_size=3,
                    padding = 1,
                    stride = 1,
                    include_original=False,
                    include_time=False).double()
        #Non-linearity
        self.tanh1 = nn.Tanh()
        #Signature with lift
        self.signature1 = signatory.LogSignature(depth=sig_d, stream=True)
        input_dim_lstm = signatory.logsignature_channels(output_channels, sig_d)
        
        #Signatures and LSTMs for signature windows
        self.lstm_sig1 = nn.LSTM(input_size=input_dim_lstm, hidden_size=hidden_dim_lstm[-2], num_layers=1, batch_first=True, bidirectional=False).double()
        self.signature2 = signatory.LogSignature(depth=sig_d, stream=False)

        input_dim_lstmsig = signatory.logsignature_channels(hidden_dim_lstm[-2] ,sig_d)
        self.lstm_sig2 = nn.LSTM(input_size=input_dim_lstmsig, hidden_size=hidden_dim_lstm[-1], num_layers=1, batch_first=True, bidirectional=True).double()

        # combination method
        if comb_method=='concatenation':
            # input dimensions for FFN
            input_dim = hidden_dim_lstm[-1] + self.embedding_dim + self.num_time_features
        elif comb_method=='gated_addition':
            input_gated_linear = hidden_dim_lstm[-1] + self.num_time_features
            if self.embedding_dim > 0:
                self.fc_scale = nn.Linear(input_gated_linear, self.embedding_dim)
                self.scaler = torch.nn.Parameter(torch.zeros(1, self.embedding_dim))
                #input dimensions for FFN
                input_dim = self.embedding_dim
            else:
                self.fc_scale = nn.Linear(input_gated_linear, input_gated_linear)
                self.scaler = torch.nn.Parameter(torch.zeros(1, input_gated_linear))
                #input dimensions for FFN
                input_dim = input_gated_linear
            # non-linearity
            self.tanh = nn.Tanh()
        elif comb_method=='gated_concatenation':
            # input dimensions for FFN
            input_dim = hidden_dim_lstm[-1] + self.embedding_dim + self.num_time_features
            # define the scaler parameter
            input_gated_linear = hidden_dim_lstm[-1] + self.num_time_features
            self.scaler1 = torch.nn.Parameter(torch.zeros(1,input_gated_linear))
        elif comb_method=='scaled_concatenation':
            input_dim = hidden_dim_lstm[-1] + self.embedding_dim + self.num_time_features
            # define the scaler parameter
            self.scaler2 = torch.nn.Parameter(torch.tensor([0.0]))

        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu1 = nn.ReLU()
        #Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # Linear function 2: 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()
        # Linear function 3 (readout): 
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def _unit_deepsignet(self,u):  
        if self.attention:
            #self attention
            out = torch.transpose(u, 0,1)
            out = torch.transpose(out, 0,2)
            out_att = self.self_attn(out, out, out)
            out = torch.transpose(out_att[0], 0, 1)
            out = torch.transpose(out, 1,2)
        else:
            out = u

        #Convolution
        if (self.augmentation_tp == 'Conv1d'):
            out = self.conv(out) #get only the path information
            out = self.tanh1(out)
            out = torch.transpose(out, 1,2) #swap dimensions
        elif (self.augmentation_tp == "CNN"):
            out = self.augment(torch.transpose(out,1,2))

        #Signature
        out = self.signature1(out)
        out, (_, _) = self.lstm_sig1(out)
        #Signature
        out = self.signature2(out)
        return out

    def forward(self, x):
        #deepsig net for each history window
        out = self._unit_deepsignet(x[:,:self.input_channels, :, -1])
        out = out.unsqueeze(1)
        for window in range(x.shape[3]-1,0, -1):
            out_unit = self._unit_deepsignet(x[:,:self.input_channels, :, window-1])
            out_unit = out_unit.unsqueeze(1)
            out = torch.cat((out, out_unit), dim=1)

        #LSTM that combines all deepsignet windows together
        _, (out, _) = self.lstm_sig2(out)
        out = out[-1, :, :] + out[-2, :, :]

        #Combine Last Post Embedding
        if self.comb_method=='concatenation':
            out = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0], x[:,(self.input_channels+self.num_time_features):, 0, 0]), dim=1)
        elif self.comb_method=='gated_addition':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0]), dim=1)
            out_gated = self.fc_scale(out_gated.float())
            out_gated = self.tanh1(out_gated)
            out_gated = torch.mul(self.scaler, out_gated)
            #concatenation with bert output
            out = out_gated + x[:,(self.input_channels+1):, 0]
        elif self.comb_method=='gated_concatenation':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0]), dim=1)
            out_gated = torch.mul(self.scaler1, out_gated)  
            #concatenation with bert output
            out = torch.cat((out_gated, x[:,(self.input_channels+self.num_time_features):, 0, 0]), dim=1 ) 
        elif self.comb_method=='scaled_concatenation':
            out_gated = torch.cat((out, x[:, self.input_channels:(self.input_channels+self.num_time_features),:, 0].max(2)[0]), dim=1)
            out_gated = self.scaler2 * out_gated  
            #concatenation with bert output
            out = torch.cat((out_gated , x[:,(self.input_channels+self.num_time_features):, 0, 0]) , dim=1 ) 

        #FFN: Linear function 1
        out = self.fc1(out.float())
        # Non-linearity 1
        out = self.relu1(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)
        #Dropout
        out = self.dropout(out)

        #FFN: Linear function 3
        out = self.fc3(out)
        return out
        