import sys
sys.path.append('../')

sys.path.append('../Models/')
sys.path.append('../Dataset/')
sys.path.append('../Utils/')

import argparse, json
from utils import *
from encoder import Encoder
from Attention import *
from DecoderWAttention import *
from XRayDataset import *
from setupEnvironment import *
from argParser import *
from TrainingEnvironment import *


import torch
import os
import numpy as np

import json
from PIL import Image
import re
import pickle
import time
from datetime import datetime


import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
frsom torch.nn.utils.rnn import pack_padded_sequence
as
from nlgeval import NLGEval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISABLE_TEACHER_FORCING = 0

def validate(argParser, val_loader, encoder, decoder, criterion, idx2word, word_map, embeddings):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """

    decoder.set_teacher_forcing_usage(DISABLE_TEACHER_FORCING)
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    references.append([])
    hypotheses = list()  # hypotheses (predictions)


    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            decoder_output, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            decoder_output_copy = decoder_output.clone()
            decoder_output = pack_padded_sequence(decoder_output, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            if argParser.model == 'Continuous':
                targets = getEmbeddingsOfTargets(targets, idx2word, word_map)
                preds = decoder_output.data #continuous model outputs prediction vector directly
                if argParser.normalizeEmb:
                    targets = torch.nn.functional.normalize(targets, p=2, dim=1)
                    preds = torch.nn.functional.normalize(preds, p=2, dim=1)
                loss = criterion(preds, targets)

            elif argParser.model == 'Softmax':
                scores  = decoder_output.data #softmax model outputs scores o probability
                targets = targets.data
                loss = criterion(scores, targets)


            # Add doubly stochastic attention regularization
            loss += argParser.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()


            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            #top5 = accuracy(scores, targets, 5)
            #top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % argParser.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses))

            # Store references (true captions), and hypothesis (prediction) for each image
            # references = [[ref1, ref2, ...]], hypotheses = [hyp1, hyp2, ...]

            # References
            temp_refs = []
            caps_sortedList = caps_sorted[:, 1:].tolist()
            for j,refCaption in enumerate(caps_sortedList):
              temp_refs.append(refCaption[:decode_lengths[j]])


            for caption in temp_refs:
              references[0].append(decodeCaption(caption, idx2word))


            # Hypotheses
            if argParser.model == 'Continuous':
              # Again, scores are the actual predicted embeddings the name is just to
              # keep inline with the softmax model... Not the best
              batch_hypotheses = generatePredictedCaptions(decoder_output_copy, decode_lengths, embeddings, idx2word)
              hypotheses.extend(batch_hypotheses)


            elif argParser.model == 'Softmax':
              _, preds = torch.max(decoder_output_copy, dim=2)
              preds = preds.tolist()
              temp_preds = list()
              for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
              preds = temp_preds

              for caption in preds:
                hypotheses.append(decodeCaption(caption, idx2word))



#            print('\nREFS: ',references[0])
#            print('\nHIPS: ',hypotheses)


            assert len(references[0]) == len(hypotheses)
 #           break


    path = '../Experiments/'+ argParser.model_name + '/valLosses.txt'
    writeLossToFile(losses.avg, path)

    return references, hypotheses, losses.avg



def train(argParser, train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, idx2word, word_map):
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
        imgs = encoder(imgs)
        decoder_output, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        decoder_output = pack_padded_sequence(decoder_output, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        if argParser.model == 'Continuous':
            targets = getEmbeddingsOfTargets(targets, idx2word, word_map)
            preds = decoder_output.data #continuous model outputs prediction vector directly
            if argParser.normalizeEmb:
                targets = torch.nn.functional.normalize(targets, p=2, dim=1)
                preds = torch.nn.functional.normalize(preds, p=2, dim=1)
            loss = criterion(preds, targets)

        elif argParser.model == 'Softmax':
            scores  = decoder_output.data #softmax model outputs scores o probability
            targets = targets.data
            loss = criterion(scores, targets)


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

def main():

    print("Starting training process MIMIC")

    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

    argParser = get_args()

    if not os.path.isdir('../Experiments/' + argParser.model_name):
        os.mkdir('../Experiments/' + argParser.model_name)

    trainingEnvironment = TrainingEnvironment(argParser)

    encoder, decoder, criterion, embeddings, word_map, encoder_optimizer, decoder_optimizer, idx2word, word2idx = setupModel(argParser)
    embeddings = embeddings.to(device)

    # Create data loaders
    trainLoader, valLoader = setupDataLoaders(argParser)

    # Load word <-> embeddings matrix index correspondence dictionaries
#    idx2word, word2idx = loadWordIndexDicts(argParser)

    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


    for epoch in range(trainingEnvironment.start_epoch, trainingEnvironment.epochs):
        # Decay learning rate if there is no improvement for "decay_LR_epoch_threshold" consecutive epochs,
        #  and terminate training after minimum LR has been achieved and  "early_stop_epoch_threshold" epochs without improvement
        if trainingEnvironment.epochs_since_improvement == argParser.early_stop_epoch_threshold:
            break
        if trainingEnvironment.epochs_since_improvement >= argParser.decay_LR_epoch_threshold and trainingEnvironment.current_lr > argParser.lr_threshold:
            trainingEnvironment.current_lr = adjust_learning_rate(decoder_optimizer, 0.5)
            if argParser.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.5)

        # One epoch's training
        train(argParser,train_loader=trainLoader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              idx2word=idx2word,
              word_map=word_map)

        # One epoch's validation
        references, hypotheses, recent_loss = validate(argParser,val_loader=valLoader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                idx2word=idx2word,
                                word_map=word_map,
                                embeddings=embeddings)

        # nlgeval = NLGEval()
        metrics_dict = nlgeval.compute_metrics(references, hypotheses)

        print("Metrics: " , metrics_dict)
        with open('../Experiments/' + argParser.model_name + "/metrics.txt", "a+") as file:
            file.write("Epoch " + str(epoch) + " results:\n")
            for metric in metrics_dict:
                file.write(metric + ":" + str(metrics_dict[metric]) + "\n")
            file.write("------------------------------------------\n")

        recent_bleu4 = metrics_dict['Bleu_4']

        # Check if there was an improvement
        is_best = recent_loss < trainingEnvironment.best_loss

        trainingEnvironment.best_loss = min(recent_loss, trainingEnvironment.best_loss)

        print("Best BLEU: ", trainingEnvironment.best_bleu4)
        if not is_best:
            trainingEnvironment.epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (trainingEnvironment.epochs_since_improvement,))
        else:
            trainingEnvironment.epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(argParser.model_name, epoch, trainingEnvironment.epochs_since_improvement, encoder.state_dict(), decoder.state_dict(), encoder_optimizer.state_dict(),
                        decoder_optimizer.state_dict(), recent_bleu4, is_best, metrics_dict, trainingEnvironment.best_loss)





if __name__ == "__main__":
    main()


