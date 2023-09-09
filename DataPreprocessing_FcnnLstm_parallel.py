import torch
import statistics
from torch import nn
from d2l import torch as d2l
torch.set_default_tensor_type(torch.DoubleTensor)


def data_preprocessing(enc_x, device):
    # Shape of enc_x: ('batch_size', 'PRN_size', 'input_size')
    # If a satellite is visible at the current epoch, this satellite will be counted.
    # Shape of valid_prn_index: (batch_size, PRN_size)
    valid_prn_index = enc_x[:, :, 1] != 0
    
    # Features of inputs: 
    # Sequence of post-preprocessing inputs
    post_enc_x_seq = []

    # 0. CN0 is scaled by 50 dBHz
    # Shape of CN0: ('batch_size', 'PRN_size', 1)
    post_enc_x_seq.append(enc_x[:, :, 8].unsqueeze(-1) / 50)

    # 1. sin and 2. cos of elevation 
    # 3. sin and 4. cos of azimuth 
    # Shape of sinE: ('batch_size', 'PRN_size', 1)
    post_enc_x_seq.append(torch.sin(enc_x[:, :, 6]).unsqueeze(-1))
    post_enc_x_seq.append(torch.cos(enc_x[:, :, 6]).unsqueeze(-1))
    post_enc_x_seq.append(torch.sin(enc_x[:, :, 32]).unsqueeze(-1))
    post_enc_x_seq.append(torch.cos(enc_x[:, :, 32]).unsqueeze(-1))

    # 5. Compute pseudorange residuals
    # Satellite positions: ('batch_size', 'PRN_size', 3)
    svXyz = enc_x[:, :, 2:5]

    # WLS-based position estimation: ('batch_size', 'PRN_size', 3)
    wlsXyz = enc_x[:, :, 10:13]

    # WLS-based user clock estimation: ('batch_size', 'PRN_size', 1)
    wlsDtu = enc_x[:, :, 13:14]

    # Pseudorange residuals: ('batch_size', 'PRN_size', 1)
    # prResi = Pr - Atmospheric Delays + Satellite Clock Bias - Geometry Range from SV to User - User Clock Bias
    prResi = enc_x[:, :, 9:10] - enc_x[:, :, 7:8] + enc_x[:, :, 5:6] - torch.linalg.vector_norm(svXyz-wlsXyz, dim=-1, keepdim=True)-wlsDtu
    post_enc_x_seq.append(prResi)

    # 6 RSS: ('batch_size', 'PRN_size', 1)
    RSS = torch.linalg.vector_norm(prResi, dim = 1, keepdim=True).repeat([1, prResi.size(dim=1), 1])
    post_enc_x_seq.append(RSS)

    # Label: ('batch_size', 'PRN_size', 1)
    # Unsmoothed Pseudorange errors
    label_enc_x = enc_x[:, :, 31:32]
        
    # Concatenate all input features
    # Shape of post_enc_x: (batch_size, PRN_size, input_feature_size)
    # Shape of valid_prn_index: (batch_size, PRN_size)
    # Shape of label_enc_x: ('batch_size', 'PRN_size', 1)
    return torch.cat(post_enc_x_seq, dim=-1), valid_prn_index, label_enc_x


