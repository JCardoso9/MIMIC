import json, random
import numpy as np
import torch
from torch import nn
import torchvision
from Attention import Attention
from BaseHierarchicalDecoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# REMOVE decoder_dim


class HierarchicalSoftmaxDecoder(BaseHierarchicalDecoder):
    """
    Decoder with continuous Outputs.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, sos_embedding,  nr_labels = 28, hidden_dim = 512, encoder_dim=1024,
                 dropout=0.5, use_tf_as_input = 1, use_scheduled_sampling=False , scheduled_sampling_prob = 0.):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(HierarchicalSoftmaxDecoder, self).__init__(attention_dim, embed_dim,decoder_dim,  vocab_size, sos_embedding, nr_labels, hidden_dim, encoder_dim,
                 dropout, use_tf_as_input, use_scheduled_sampling , scheduled_sampling_prob)


        
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution



    def forward(self, encoder_out, encoded_captions, caption_lengths, labels):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)




        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        input = self.sos_embedding.expand(batch_size, 200).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([input[:batch_size_t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            # When not using teacher forcing or with scheduled sampling prob
            # use the embedding for the previous generated word
            if self.use_tf_as_input == 0 or self.use_scheduled_sampling and random.random() < self.scheduled_sampling_prob:
                _, preds = torch.max(preds, dim=1)
                input = self.embedding(preds)



            # Else, use teacher forcing and provide words from reference
            else:
                if t <= max(decode_lengths) -1 :
                    input = embeddings[:batch_size_t, t+1, :]





        return predictions, encoded_captions, decode_lengths, alphas, sort_ind



