import torch
from torch import nn
import torchvision
import math

class DotProdAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(DotProdAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.k_linear =  nn.Linear(encoder_dim, encoder_dim)
        self.v_linear =  nn.Linear(encoder_dim, encoder_dim)
        self.q_linear =  nn.Linear(decoder_dim, encoder_dim)



    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        k = self.k_linear(encoder_out)
        print("k shape", k.shape)

        v = self.v_linear(encoder_out)
        print("v shape", v.shape)
        q = self.q_linear(decoder_hidden)
        print("q shape", q.shape)

        d_k = encoder_out.shape[2]
        print("Q uns",q.unsqueeze(1).shape)

        alphas = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) /  math.sqrt(d_k)
        print(alphas.shape)
        alphas = self.softmax(alphas)
        awe = torch.matmul(alphas, v).squeeze(1)
        print("awe", awe.shape)



        return attention_weighted_encoding, alphas



