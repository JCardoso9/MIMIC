
import sys
sys.path.append('../Dataset/')
sys.path.append('../Models/')
sys.path.append('../Utils/')
sys.path.append('../')

from Encoder import Encoder
from Attention import *
from SoftmaxDecoder import *
from ContinuousDecoder import *
from XRayDataset import *
from TrainingEnvironment import *
from losses import *

from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

import json
from PIL import Image
import re
from torchvision import transforms
import pickle


import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setupModel(args):
  '''
    Setup the model according to the mode being used (Training/Testing), 
    loading checkpoints when necessary.
    :param args: Argument Parser object with definitions set in a specified config file
  '''

  # Load embeddings from disk
  word_map, embeddings, vocab_size, embed_dim = loadEmbeddingsFromDisk(args.embeddingsPath, args.normalizeEmb)
  embeddings = embeddings.to(device)

  idx2word, word2idx = loadWordIndexDicts(args)

  # Create adequate model
  if (args.model == 'Continuous'):
    decoder = ContinuousDecoder(attention_dim=args.attention_dim,
                                    embed_dim=embed_dim,
                                    decoder_dim=args.decoder_dim,
                                    vocab_size=vocab_size,
                                    sos_embedding = embeddings[word2idx['<sos>']],
                                    dropout=args.dropout,
                                    use_tf_as_input = args.use_tf_as_input,
                                    use_scheduled_sampling=args.use_scheduled_sampling,
                                    scheduled_sampling_prob=args.initial_scheduled_sampling_prob,
                                    use_custom_tf=args.use_custom_tf)

  elif (args.model == 'Softmax'):
    decoder = SoftmaxDecoder(attention_dim=args.attention_dim,
                                    embed_dim=embed_dim,
                                    decoder_dim=args.decoder_dim,
                                    vocab_size=vocab_size,
                                    sos_embedding = embeddings[word2idx['<sos>']],
                                    dropout=args.dropout,
                                    use_tf_as_input = args.use_tf_as_input,
                                    use_scheduled_sampling=args.use_scheduled_sampling,
                                    scheduled_sampling_prob=args.initial_scheduled_sampling_prob)



  decoder.load_pretrained_embeddings(embeddings)
  encoder = Encoder()

  # Load trained model if checkpoint exists
  if (args.checkpoint is not None):
    modelInfo = torch.load(args.checkpoint)
    decoder.load_state_dict(modelInfo['decoder'])
    encoder.load_state_dict(modelInfo['encoder'])

  # Move to GPU, if available
  decoder = decoder.to(device)
  encoder = encoder.to(device)

  # Set model to eval mode
  if (args.runType == "Testing"):
    decoder.eval()
    encoder.eval()
    encoder_optimizer = None
    decoder_optimizer = None
    enc_scheduler = None
    dec_scheduler = None

  # If training, create optimizers. If necessary, load previous checkpoint.
  # Also check if fine tuning embeddings and/or encoder is necessary
  elif (args.runType == "Training"):
    decoder.fine_tune_embeddings(args.fine_tune_embeddings)
    encoder.fine_tune(args.fine_tune_encoder)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                          lr=args.decoder_lr)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=args.encoder_lr) if args.fine_tune_encoder else None
    if (args.checkpoint is not None):
      decoder_optimizer.load_state_dict(modelInfo['decoder_optimizer'])
      encoder_optimizer.load_state_dict(modelInfo['encoder_optimizer'])

    dec_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, args.lr_decay_epochs, args.lr_decay)
    enc_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, args.lr_decay_epochs, args.lr_decay)

  # Create criterion
  if (args.loss == 'CrossEntropy'):
    criterion = nn.CrossEntropyLoss().to(device)

  elif (args.loss == 'CosineSimilarity'):
    criterion = CosineEmbedLoss()

  elif(args.loss == 'SmoothL1'):
    criterion = SmoothL1LossWord().to(device)

  elif(args.loss == 'SmoothL1WordAndSentence'):
    criterion = SmoothL1LossWordAndSentence().to(device)

  elif (args.loss == 'TripleMarginLoss'):
    criterion = SyntheticTripletLoss(args.triplet_loss_margin, args.triplet_loss_mode)

  return encoder, decoder, criterion, embeddings, encoder_optimizer, decoder_optimizer, dec_scheduler, enc_scheduler, idx2word, word2idx







def setupDataLoaders(args):
  """
    Create the necessary data loaders according to run type: Training/Testing
    :param args: Argument Parser object with definitions set in a specified config file
  """

  # Transforms required for models pre-trained o ImageNet
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

  # Create MIMIC dataset loaders
  if (args.runType == "Testing"):
    testLoader = DataLoader(XRayDataset(args.word2idxPath, args.encodedTestCaptionsPath,
      args.encodedTestCaptionsLengthsPath, args.testImgsPath, transform), batch_size=args.batch_size, shuffle=True)

    return testLoader, None

  elif (args.runType == "Training"):
    trainLoader = DataLoader(XRayDataset(args.word2idxPath, args.encodedTrainCaptionsPath,
      args.encodedTrainCaptionsLengthsPath, args.trainImgsPath, transform), batch_size=args.batch_size, shuffle=True)
    valLoader = DataLoader(XRayDataset(args.word2idxPath, args.encodedValCaptionsPath,
      args.encodedValCaptionsLengthsPath, args.valImgsPath, transform), batch_size=1, shuffle=True)

    return trainLoader, valLoader



def initializeTrainingEnvironment(args):
  return TrainingEnvironment(args)


def loadWordIndexDicts(args):
  '''
    Load the dictionaries with the Word <-> embeddings matrix index correspondence
    :param args: Argument Parser object with definitions set in a specified config file
  '''
  with open(args.idx2wordPath) as fp:
        idx2word = json.load(fp)
  with open(args.word2idxPath) as fp:
        word2idx = json.load(fp)
  return idx2word, word2idx




def loadEmbeddingsFromDisk(embeddingsPath, normalize=True):
  """
    Load the dictionary with word -> embeddings correspondence. Return also vocab size, embeddings matrix and
    the embeddings dimension
    :param embeddingsPath: Path to pkl object with the dictionary.
  """
  word_map = pickle.load(open(embeddingsPath, "rb" ))
  vocab_size = len(list(word_map.keys()))
  embed_dim = len(list(word_map.values())[1])
  embeddings = torch.FloatTensor(list(word_map.values()))

  if normalize:
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    words = list(word_map.keys())
    for n in range(len(words)):
      word_map[words[n]] = embeddings[n].numpy()

  return word_map, embeddings, vocab_size, embed_dim
