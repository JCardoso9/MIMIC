import torch
from torch import nn
import torchvision


class ClassAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, nr_labels, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(ClassAttention, self).__init__()
        self.encoder_att = nn.Linear(1, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, labels, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        print("labels", labels.unsqueeze(2).shape)
        att1 = self.encoder_att(labels.unsqueeze(2))  # (batch_size, num_pixels, attention_dim)
        print("LABELS SHAPE",att1.shape)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        print(alpha.shape)
        attention_weighted_encoding = (labels * alpha)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
