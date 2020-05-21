import json
import torch
from torch import nn
import torchvision
from Attention import Attention
from abc import ABC, abstractmethod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseDecoderWAttention(nn.Module):
    """
    Decoder with continuous Outputs.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, sos_embedding, nr_labels, hidden_dim = 256, encoder_dim=1024,
                 dropout=0.5, use_tf_as_input = 1, use_scheduled_sampling=False , scheduled_sampling_prob = 0.):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(BaseDecoderWAttention, self).__init__()

        self.nr_labels = nr_labels
        self.img_encoding_dim = img_encoding_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.sos_embedding = sos_embedding
        self.use_tf_as_input = use_tf_as_input
        self.use_scheduled_sampling = use_scheduled_sampling
        self.scheduled_sampling_prob = scheduled_sampling_prob
        self.hidden_dim = hidden_dim


        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        
        self.resize_encoder_features = nn.Linear(encoder_dim, hidden_dim)
        
        self.sentence_decoder = nn.LSTMCell(hidden_dim + nr_labels, decoder_dim, bias=True)  # decoding LSTMCell
        self.word_decoder = nn.LSTMCell(embed_dim + hidden_dim, decoder_dim, bias=True)

        self.init_h_sentence_dec = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_sentence_dec = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.init_h_word_dec = nn.Linear(encoder_features_resized_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_word_dec = nn.Linear(encoder_features_resized_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        
        self.sentence_decoder_fc_sizes = [
            self.hidden_size,  # topic
            1                 # stop
        ]
        self.sentence_decoder_fc = nn.Linear(self.hidden_size, sum(self.sentence_decoder))

        self.topic_vector = nn.Linear(decoder_dim, 
        self.stop_signal = nn.Linear(decoder_dim, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
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
        h_sent = self.init_h_sentence_dec(mean_encoder_out)  # (batch_size, decoder_dim)
        c_sent = self.init_c_sentence_dec(mean_encoder_out)
        h_word = self.init_h_sentence_dec(mean_encoder_out)  # (batch_size, decoder_dim)
        c_word = self.init_c_sentence_dec(mean_encoder_out)
        return h_sent, c_sent, h_word, c_word

    def set_teacher_forcing_usage(self, value):
        self.use_tf_as_input = value

    def set_scheduled_sampling_usage(self, value):
        self.use_scheduled_sampling = value

    def set_scheduled_sampling_prob(self, value):
        self.schedule_sampling_prob = value

    def should_use_prev_output():
        return random.random() < self.scheduled_sampling_prob

    @abstractmethod
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        pass
