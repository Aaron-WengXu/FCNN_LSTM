import torch
from torch.nn.utils.rnn import pack_sequence
torch.set_default_tensor_type(torch.DoubleTensor)

def data_random_pack(post_enc_x, valid_prn_index):
    # Shape of post_enc_x: ('batch_size', 'PRN_size', 'input_feature_size')
    # Shape of valid_prn_index: (batch_size, PRN_size)
    valid_post_enc_x_list = []
    for (eachBatch, eachValid) in zip(post_enc_x, valid_prn_index):
        # Shape of eachBatch: ('PRN_size', 'input_feature_size')
        # Shape of eachValid: (PRN_size)
        # Shape of valid_post_enc_x: ('PRN_valid_size', 'input_feature_size')
        valid_post_enc_x = eachBatch[eachValid]

        # Shuffle the satellite dimension
        shuffled_index = torch.randperm(valid_post_enc_x.size(dim=0))
        valid_post_enc_x_shuffled = valid_post_enc_x[shuffled_index, :]

        valid_post_enc_x_list.append(valid_post_enc_x_shuffled)
    return pack_sequence(valid_post_enc_x_list, enforce_sorted=False)

