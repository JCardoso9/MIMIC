import json
import torch
from torch import nn
import torchvision
from Attention import Attention
from DotProdAttention import *
from abc import ABC, abstractmethod
from MogrifierLSTM import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDecoderWAttention(nn.Module):
    """
    Decoder with continuous Outputs.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, sos_embedding, encoder_dim, 
                 dropout, use_tf_as_input, use_scheduled_sampling , scheduled_sampling_prob, use_mogrifier,
                 attention_type):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        #encoder_dim = 1024
        #print(encoder_dim)
        super(BaseDecoderWAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.sos_embedding = sos_embedding
        self.use_tf_as_input = use_tf_as_input
        self.use_scheduled_sampling = use_scheduled_sampling
        self.scheduled_sampling_prob = scheduled_sampling_prob
        print("Decoder Initial SS prob",scheduled_sampling_prob)
        self.use_mogrifier = use_mogrifier
        self.attention_type = attention_type
        print("Attention",self.attention_type)
        if self.attention_type == "Additive":
            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        elif self.attention_type == "DotProduct":
            print("Dot product attention")
            self.attention = DotProdAttention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        if (use_mogrifier):
            print("created mogrifier")
            self.decode_step =  MogrifierLSTMCell(embed_dim+encoder_dim, decoder_dim, 3)
        else:
            self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        
        #encoder_dim = 1024
        #print(encoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        #self.fc = nn.Linear(decoder_dim, embed_dim)  # linear layer to generate continuous outputs
        #self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def set_teacher_forcing_usage(self, value):
        self.use_tf_as_input = value

    def set_scheduled_sampling_usage(self, value):
        self.use_scheduled_sampling = value

    def set_scheduled_sampling_prob(self, value):
        print("new Ss", value)
        self.schedule_sampling_prob = value
        print("decoder prob:", self.schedule_sampling_prob)

    def should_use_prev_output():
        return random.random() < self.scheduled_sampling_prob

    @abstractmethod
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        pass
