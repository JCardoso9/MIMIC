import argparse, json
from utils import *
from encoder import Encoder
from Attention import *
from DecoderWAttention import *
from XRayDataset import *

import torch
import os
import numpy as np

import json
from PIL import Image
import re
import torch.nn as nn
import pickle
import time
from datetime import datetime


import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import SmoothingFunction

from nlgeval import NLGEval





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
start_epoch = 0
epochs = 20  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 16
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 5  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none



def validate(idx2word, val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
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
    # solves the issue #57
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
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            #top5 = accuracy(scores, targets, 5)
            #top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses))
                
            

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            # for j in range(allcaps.shape[0]):
            #     img_caps = allcaps[j].tolist()
            #     img_captions = list(
            #         map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
            #             img_caps))  # remove <start> and pads
            #     references.append(img_captions)

            

            temp_refs = []
            caps_sortedList = caps_sorted[:, 1:].tolist()
            # print("Caps sorted: ", caps_sorted.shape)
            for j,refCaption in enumerate(caps_sortedList):
              # print("J", j)
              # print("i", i)

              temp_refs.append(refCaption[:decode_lengths[j]]) 

            # print("-------")
            decodedTempRef = []

            for caption in temp_refs:
              # print("Length ref: ", len(caption))
              # print("REF: ", caption)
              caption = [str(i) for i in caption]
              decoded = decodeCaption(caption, idx2word)
              # print("Length ref decoded: ", len(decoded))
              # decodedTempRef.append(createHypothesis(decoded))
              #print(decodedTempRef)
              references[0].append(createHypothesis(decoded))

            # print("Caps sorted: ", caps_sorted.shape)
            # print("References:", len(references))
            # print("Full references: ", references)

                      

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds

            decodedTempRef = []
            
            for caption in preds:
              # print("Length preds: ", len(caption))
              # print("PRED: ", caption)
              caption = [str(i) for i in caption]
              decoded = decodeCaption(caption, idx2word)
              # print("Length preds decoded: ", len(decoded))
              
              decodedTempRef.append(createHypothesis(decoded))
            #print(decodedTempRef)
            hypotheses.extend(decodedTempRef)
            
            # print("-------")



            # print("--------------")
            # print("hypothesis: ", len(hypotheses))
            # print("Full hypothesis: ", hypotheses)
          
            
            # print(references[0])
            # print(hypotheses)

            

            assert len(references[0]) == len(hypotheses)
            # break

        # Calculate BLEU-4 scores
        #bleu4 = corpus_bleu(references, hypotheses)
        # nlgeval = NLGEval()  # loads the models
        # metrics_dict = nlgeval.compute_metrics(references, hypotheses)

        # print(
        #     '\n * LOSS - {loss.avg:.3f}, BLEU-4 - {bleu}\n'.format(
        #         loss=losses,
        #         bleu=bleu4))
    now = datetime.now()
    day_string = now.strftime("%d_%m_%Y")
    path = 'valLosses_' + day_string
    writeLossToFile(losses.avg, path)

    return references, hypotheses



def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
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
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # print("Caps sorted: ", caps_sorted.shape)
        # print("Decode Lengths: ", decode_lengths)
        # print("Scores: ", scores.shape)
        # print("Targets: ", targets.shape)
        # print("Decode Lengths: ", decode_lengths)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # print("After")
        # print("Scores: ", scores.data.shape)
        # print("Targets: ", targets.data.shape)

        # Calculate loss
        loss = criterion(scores.data, targets.data)

        # print("LOSS: " ,loss)
        #break

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        # if grad_clip is not None:
        #     clip_gradient(decoder_optimizer, grad_clip)
        #     if encoder_optimizer is not None:
        #         clip_gradient(encoder_optimizer, grad_clip)

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
        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))
    now = datetime.now()
    day_string = now.strftime("%d_%m_%Y")
    path = 'trainLosses_' + day_string
    writeLossToFile(losses.avg, path)

def main(checkpoint=None):

    print("Starting training process MIMIC")
    global device, best_bleu4, epochs_since_improvement, start_epoch, fine_tune_encoder

    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

    word_map, embeddings, vocab_size, embed_dim = loadEmbeddingsFromDisk('/home/jcardoso/MIMIC/embeddingsMIMIC.pkl')

    with open('/home/jcardoso/MIMIC/idx2word.json') as fp:
        idx2word = json.load(fp)


    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                        embed_dim=embed_dim,
                                        decoder_dim=decoder_dim,
                                        vocab_size=vocab_size,
                                        dropout=dropout)

        decoder.load_pretrained_embeddings(embeddings)  # pretrained_embeddings should be of dimensions (len(word_map), emb_dim)
        decoder.fine_tune_embeddings(False)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                              lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                              lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # Custom dataloaders
    trainLoader = DataLoader(
    XRayDataset("/home/jcardoso/MIMIC/word2idx.json","/home/jcardoso/MIMIC/encodedTestCaptions.json",'/home/jcardoso/MIMIC/encodedTestCaptionsLengths.json','/home/jcardoso/MIMIC/Test', transform),
     batch_size=4, shuffle=True)


    valLoader = DataLoader(
        XRayDataset("/home/jcardoso/MIMIC/word2idx.json","/home/jcardoso/MIMIC/encodedValCaptions.json",'/home/jcardoso/MIMIC/encodedValCaptionsLengths.json','/home/jcardoso/MIMIC/Val', transform),
         batch_size=4, shuffle=True)


    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=trainLoader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        references, hypotheses = validate(idx2word, val_loader=valLoader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # nlgeval = NLGEval()
        metrics_dict = nlgeval.compute_metrics(references, hypotheses)

        recent_bleu4 = metrics_dict['Bleu_4']

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4

        best_bleu4 = max(recent_bleu4, best_bleu4)

        print("Best BLEU: ", best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint( epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, metrics_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--checkpoint', type=str, default='', metavar='N',
                        help='Path to the model\'s checkpoint (No checkpoint: empty string)')


    args = parser.parse_args()
    if args.checkpoint:
        main(args.checkpoint)

    else:
        main()
