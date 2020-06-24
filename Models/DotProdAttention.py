import torch
from torch import nn
import torchvision


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
        self.k_linear =  nn.Linear(encoder_dim, attention_dim)
        self.v_linear =  nn.Linear(encoder_dim, attention_dim)
        self.q_linear =  nn.Linear(decoder_dim, attention_dim)



    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        k = self.linear(encoder_out)
        v = self.linear(encoder_out)
        q = self.linear(encoder_out)
        d_k = decoder_hidden.shape[1]

        alphas = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        alphas = self.softmax(alphas)
        awe = torch.matmul(alphas, v)


        return attention_weighted_encoding, alpha



