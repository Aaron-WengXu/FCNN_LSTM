import torch
import time
import statistics
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l
from DataPreprocessing_FcnnLstm_parallel import data_preprocessing
from Data_randomizing_packing import data_random_pack
torch.set_default_tensor_type(torch.DoubleTensor)
torch.autograd.set_detect_anomaly(True)


def train_fcnn_lstm(net, data_iter, lr, num_epochs, num_iterations, device):
    """ Train a model with FCNN-LSTM """

    # Delegate computation to CPU or GPU
    net.to(device)

    # Determine the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Define loss function
    loss = nn.MSELoss()

    # Set the neural network to training mode
    net.train()

    # Set figure to plot training loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, 1000])
    
    time_step = 0 

    # Evaluate training time
    time_sum = []

    # Training epoch by epoch
    for epoch in range(num_epochs):
        for batch in data_iter:           
            optimizer.zero_grad()
            start_time = time.time()
            for iteration_step in range(num_iterations):
                time_step = time_step + 1

                # Read a batch of training data and delegate the data to our device
                # Shape of X: (batch_size, PRN_size, input_size)
                X, _ = [x.to(device) for x in batch]
               
                # Shape of enc_x: (batch_size, PRN_size, input_size)
                enc_x = X

                # Data Preprocessing
                # Shape of post_enc_x:       ('batch_size', 'PRN_size', 'input_feature_size')
                # Shape of label_enc_x:      ('batch_size', 'PRN_size', '1')                  
                # Shape of valid_prn_index:  ('batch_size', 'PRN_size')
                post_enc_x, valid_prn_index, label_enc_x = data_preprocessing(enc_x, device)

                # Randomize the input on satellite dimensions and pack to sequence
                # post_enc_x_random_pack: packed sequences as input of LSTM
                post_enc_x_random_pack = data_random_pack(post_enc_x, valid_prn_index)

                # Pass input data through the neural network
                # Shape of dec_y_scaled:    ('batch_size', 'PRN_size', 1)
                # Shape of dec_y:           ('batch_size', 'PRN_size', 1)
                # Shape of total_prm_error: ('batch_size', 'PRN_size', 1)
                total_prm_error = net(post_enc_x, post_enc_x_random_pack)
                
                # Loss Computation
                # Shape of broadcast_index_valid:  ('batch_size', 'PRN_size',1)
                broadcast_index_valid = valid_prn_index.unsqueeze(-1)

                # Compute the error of clock bias estimation                
                # Training using smoothed pseudoranges
                J = loss(total_prm_error[broadcast_index_valid], label_enc_x[broadcast_index_valid])
                
                # Backward Gradient Descent
                J.sum().backward()
                d2l.grad_clipping(net, 1)
                optimizer.step()
                # animator.add(time_step, [J.cpu().detach().numpy()])
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_sum.append(elapsed_time)

        if (epoch + 1) % 1 == 0:
            animator.add(epoch + 1, [J.cpu().detach().numpy()])
            # animator1.add(epoch + 1, [J1.cpu().detach().numpy()])
            

        # if (epoch + 1) % 1 == 0:
        #     print('Epoch', epoch+1, 'is done. ', 'Loss is',J.cpu().detach().numpy())
        # if (epoch + 1) % 100 == 0:           
        #     filename = 'PrNet_2023V_' + str(epoch+1+1900) + '.tar'       
        #     torch.save({
        #         'model_state_dict': net.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         }, filename)
    elapsed_time_per_batch = sum(time_sum)/len(time_sum)
    print('Training time per batch: ', elapsed_time_per_batch)
    return optimizer
