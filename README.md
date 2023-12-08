# FCNN_LSTM
Fully Connected Neural Network-LSTM for pseudorange correction

FCNN-LSTM is proposed in the paper "Zhang, G., Xu, P., Xu, H., & Hsu, L. T. (2021). <em>Prediction on the urban GNSS measurement uncertainty based on deep learning networks with long short-term memory</em>. IEEE Sensors Journal, 21(18), 20563-20577". This repository is our implementation of it. Please refer to the original [paper](https://ieeexplore.ieee.org/document/9490205) for more details.

Our implementation is built upon Pytorch library. And the data-preprocessing functions, training functions, and evaluation functions are properties of NTUsg. If you wanna use our codes in yours, please kindly cite us as:
[Weng, X., Ling, K. V., & Liu, H. (2023). PrNet: A Neural Network for Correcting Pseudoranges to Improve Positioning with Android Raw GNSS Measurements. arXiv preprint arXiv:2309.12204.](https://arxiv.org/abs/2309.12204)

You can start from the Jupyter Notebook "FCNN_LSTM_MultipleFile_parallel.ipynb" to explore this model. Before that, you need to set up the environment on a Linux computer using "environment.yml" we provided.  
