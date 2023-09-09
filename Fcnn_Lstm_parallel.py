from torch import nn
import torch
from d2l import torch as d2l
torch.set_default_tensor_type(torch.DoubleTensor)


# %% ********************* FCNN-LSTM Encoder  *********************
# Input shape:  batch_size * Prn_size * input_size
# Output PrM bias shape: batch_size * Prn_size * 1
class Fcnn_LSTM_encoder(nn.Module):
    """The FCNN-LSTM encoder for debiasing."""

    def __init__(self, input_size_debiasing, num_hiddens_debiasing_fcnn,
                 num_hiddens_debiasing_lstm, dropout=0, **kwargs):
        super(Fcnn_LSTM_encoder, self).__init__(**kwargs)
        # LSTM Layer
        self.lstmLayer = nn.LSTM(input_size = input_size_debiasing,
                                 hidden_size = num_hiddens_debiasing_lstm,
                                 batch_first = True)

        # Satellite-wise FCNN encoder
        self.svFcnnLayerEncoder = nn.Sequential(nn.Linear(input_size_debiasing, num_hiddens_debiasing_fcnn),
                                    nn.ReLU(),
                                    # Add a dropout layer after the first fully connected layer
                                    nn.Dropout(dropout),
                                    nn.Linear(num_hiddens_debiasing_fcnn, num_hiddens_debiasing_fcnn),
                                    nn.ReLU(),
                                    # Add a dropout layer after the second fully connected layer
                                    nn.Dropout(dropout),
                                    nn.Linear(num_hiddens_debiasing_fcnn, num_hiddens_debiasing_fcnn),
                                    nn.ReLU())

        # Satellite-wise FCNN decoder
        self.svFcnnLayerDecoder = nn.Sequential(nn.Linear(num_hiddens_debiasing_fcnn+num_hiddens_debiasing_lstm, num_hiddens_debiasing_fcnn),
                                    nn.ReLU(),
                                    # Add a dropout layer after the first fully connected layer
                                    nn.Dropout(dropout),
                                    nn.Linear(num_hiddens_debiasing_fcnn, num_hiddens_debiasing_fcnn),
                                    nn.ReLU(),
                                    # Add a dropout layer after the second fully connected layer
                                    nn.Dropout(dropout),
                                    # The output layer
                                    nn.Linear(num_hiddens_debiasing_fcnn, 1))
        
    
    def forward(self, X, X_random_pack, *args):            
        # Shape of X: ('batch_size', 'Prn_size', 'input_size')
        # Shape of X_random_pack: ('batch_size', 'Prn_valid_size', 'input_size')
        
        # %% LSTM layer
        # output is a packed sequence
        # Shape of hk: (1, 'batch_size', 'hidden_size_lstm')
        # hk have been sorted back automatically by nn.LSTM
        output, (hk, ck) = self.lstmLayer(X_random_pack)

        # %% Satellite-wise FCNN encoder
        # Shape of y: ('batch_size', 'Prn_size', 'hidden_size_fcnn')
        y = self.svFcnnLayerEncoder(X)

        # %% Hidden state concatenation
        # Shape of yhk: ('batch_size', 'Prn_size', 'hidden_size_fcnn+hidden_size_lstm')
        yhk = torch.concat([y, hk.permute(1,0,2).repeat([1, y.size(dim=1),1])], dim = -1)

        # %% Satellite-wise FCNN decoder
        # `output` shape: ('batch_size', 'Prn_size', 1)
        return self.svFcnnLayerDecoder(yhk)

           
# %%  ********************* FCNN-LSTM *********************
# @save
class Fcnn_Lstm(nn.Module):
    """The base class for FCNN-LSTM"""

    def __init__(self, debiasing_layer, **kwargs):
        super(Fcnn_Lstm, self).__init__(**kwargs)
        self.debiasing_layer = debiasing_layer

    def forward(self, feature_inputs, feature_inputs_random_pack, *args):
        prm_bias = self.debiasing_layer(feature_inputs, feature_inputs_random_pack, *args)
        # `output` shape: ('batch_size', 'Prn_size', 1)
        return prm_bias