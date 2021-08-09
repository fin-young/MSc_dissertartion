from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return silu(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class VGG(nn.Module):
    def __init__(self, vgg_name, act_fn='relu'):
        super(VGG, self).__init__()

        if act_fn == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act_fn == 'relu':
            self.act = torch.nn.ReLU()
        elif act_fn == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_fn == 'silu':
            self.act = SiLU()
        elif self.act_fn == 'elu':
            self.act = torch.nn.ELU()

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           self.act
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seq_len = seq_length

        # LSTM layers
        self.lstm = nn.LSTM(
                            input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=layer_dim,
                            batch_first=True,
                            dropout=dropout_prob
                            )

        # Fully connected layer
        #Edit from: nn.Linear(hidden_dim*self.seq_len, output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    #def init_hidden(self, batch_size):
    #    # even with batch_first = True this remains same as docs
    #    hidden_state = torch.zeros(self.layer_dim,batch_size,self.hidden_dim)
    #    cell_state = torch.zeros(self.layer_dim,batch_size,self.hidden_dim)
    #    self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        #batch_size, seq_len, _ = x.size()
        # Initializing hidden state for first input with zeros
        #Should x.size(0) be batch_size? 
        hidden_state = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        cell_state = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        lstm_out, (hn, cn) = self.lstm(x, (hidden_state.detach(), cell_state.detach()))
        #lstm_out, self.hidden = self.lstm(x,self.hidden)
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        
        lstm_out = lstm_out[:, -1, :]
        
        # Convert the final state to our desired output shape (batch_size, output_dim)
        lstm_out = self.fc(lstm_out)

        return lstm_out

def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)