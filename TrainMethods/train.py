import sys
sys.path.append('../Utils/')

from generalUtilities import *
#from continuousModelUtilities import *

import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REmove id2word and word_map

def train(argParser, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.set_teacher_forcing_usage(argParser.use_tf_as_input)
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        if (argParser.use_classifier_encoder):
            imgs, _ = encoder(imgs)
        else:
            imgs = encoder(imgs)

        decoder_output, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        if (argParser.use_img_embedding):
             targets = (caps_sorted[:, 1:], decoder_output[1]
             decoder_output = decoder_output[0]
        else:
             targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this

        # Calculate loss
        if argParser.model == 'Continuous':
            targets = decoder.embedding(targets)
            if argParser.normalizeEmb:
                targets = nn.functional.normalize(targets, p=2, dim=1)
                preds = nn.functional.normalize(decoder_output, p=2, dim=1)
            loss = criterion(preds, targets, decode_lengths)

        elif argParser.model == 'Softmax':
            decoder_output = pack_padded_sequence(decoder_output, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            loss = criterion(decoder_output.data, targets.data)


        # Add doubly stochastic attention regularization
        loss += argParser.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if argParser.grad_clip is not None:
            clip_gradient(decoder_optimizer, argParser.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, argParser.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        #top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % argParser.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))
    path = '../Experiments/' + argParser.model_name + '/trainLosses.txt'
    writeLossToFile(losses.avg, path)
