import torch
import time
from torch import nn
from DataPreprocessing_FcnnLstm_parallel import data_preprocessing
from Data_randomizing_packing import data_random_pack
from d2l import torch as d2l
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)
torch.autograd.set_detect_anomaly(True)


def evaluate_fcnnLstm(net, data_iter, batch_size, device):
    """Evaluate a model with GNSS moving horizon DAE."""

    # Delegate computation to CPU or GPU
    net.to(device)

    # Set the neural network to training mode
    net.eval()

    # Define loss function
    loss = nn.MSELoss()

    # Set figure to plot training loss
    animator = d2l.Animator(xlabel='time step', ylabel='loss')

    time_step = 0

    time_sum = []

    # Initialize the output list
    output_seq = []

    # batch size is 1
    for batch in data_iter:

        time_step = time_step+1

        # Read a batch of training data and delegate the data to our device
        # Shape of x: (batch_size, PRN_size, input_size)
        x, _ = [z.to(device) for z in batch]

        # Shape of enc_x: (batch_size, PRN_size, input_size)
        enc_x = x

        # Data Preprocessing
        # Shape of post_enc_x:       ('batch_size', 'PRN_size', 'input_feature_size')
        # Shape of label_enc_x:      ('batch_size', 'PRN_size', '1')                  
        # Shape of valid_prn_index:  ('batch_size', 'PRN_size')
        post_enc_x, valid_prn_index, label_enc_x = data_preprocessing(enc_x, device)

        # Randomize the input on satellite dimensions and pack to sequence
        # post_enc_x_random_pack: packed sequences as input of LSTM
        post_enc_x_random_pack = data_random_pack(post_enc_x, valid_prn_index)

        start_time = time.time()

        # Pass input data through the neural network
        # Shape of total_prm_error: ('batch_size', 'PRN_size', 1)
        total_prm_error = net(post_enc_x, post_enc_x_random_pack)

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_sum.append(elapsed_time)

        # Loss Computation
        # Shape of broadcast_index_valid:  ('batch_size', 'PRN_size',1)
        broadcast_index_valid = valid_prn_index.unsqueeze(-1)
                
        # Training loss using unsmoothed pseudoranges with clock estimation residuals
        J = loss(total_prm_error[broadcast_index_valid], label_enc_x[broadcast_index_valid])
        
        # Form the output of PrM bias
        for i in range(batch_size):
            # Shape of enc_x_per_batch: ('PRN_size', input_size)
            enc_x_per_batch = enc_x[i]

            # Shape of index_prn_prmbias_per_batch: (`Valid_PRN_size`, 3)
            index_prn_prmbias_per_batch = torch.cat([enc_x_per_batch[valid_prn_index[i], 0:2], total_prm_error[i,valid_prn_index[i],:]], dim=1)
            output_seq.append(index_prn_prmbias_per_batch)

        animator.add(time_step, [J.cpu().detach().numpy()])

    elapsed_time_per_sample = sum(time_sum)/len(time_sum)
    print('Inference time per sample: ', elapsed_time_per_sample)
    # Shape of 'return': (`All_Valid_PRN_size`, 3)
    return  torch.cat(output_seq, dim=0)


