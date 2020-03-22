from datetime import datetime

import torch
import torch.nn as nn
import os
import numpy as np

import json
from PIL import Image
import re
from torchvision import transforms
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt


import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def writeLossToFile(loss, path):
  """
    Write loss of the curren batch to a log file
    :param loss: Average loss of the current batch.
    :param path: path to the log file
  """
  with open(path, 'a+') as file:
    file.write(str(loss) + '\n')


    
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def getFilesInDirectory(path):
  """
    Get the path of all files in this directory, recursively
    the embeddings dimension
    :param path: Root path. 
  """
  files = []
  for r, d, f in os.walk(path):
      for file in f:
          if '.jpg' in file:
              files.append(os.path.join(r, file))
  return files


def loadEmbeddingsFromDisk(embeddingsPath):
  """
    Load the dictionary with word -> embeddings correspondence. Return also vocab size, embeddings matrix and
    the embeddings dimension
    :param embeddingsPath: Path to pkl object with the dictionary. 
  """
  word_map = pickle.load(open(embeddingsPath, "rb" ))
  vocab_size = len(list(word_map.keys()))
  embed_dim = len(list(word_map.values())[1])
  embeddings = torch.FloatTensor(list(word_map.values()))

  return word_map, embeddings, vocab_size, embed_dim


# Improve this code
def decodeCaption(caption, idx2word):
  """
    Transform captions comprised of word indexes into natural language captions
    :param caption: caption comprised of list of indexes.
    :param idx2word: dictionary with index -> word correspondence. 
  """
  lenVocabularyWords = len(idx2word.keys())
  if caption[0] > lenVocabularyWords:
    decodedCaption = '<unk>'
  else:
    decodedCaption = idx2word[str(caption[0])]
  
  for index in caption[1:-1]:
    if index > lenVocabularyWords:
      if index == 17942:
        decodedCaption += '.'
      else:
        decodedCaption += ' <unk>'
    else:
      if index == 6757:
        decodedCaption += '.'
      else:
        word = idx2word[str(index)]
        decodedCaption += ' ' + word   
      
  return decodedCaption


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))



def save_checkpoint( epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, metrics_dict):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """

    now = datetime.now()
 
    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S_")

    day_string = now.strftime("%d_%m_%Y")
    print("date and time =", dt_string)
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'metrics_dict': metrics_dict}

    filename = 'checkpoint_' + dt_string + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = 'checkpoint_' + day_string + '.pth.tar'
        torch.save(state, 'BEST_' + filename)
        
        
        
def plotLosses(trainLossesPath, valLossesPath):
  trainLosses = []
  valLosses = []
  with open(trainLossesPath) as file:
    for line in file:
      trainLosses.append(float(line))
  with open(valLossesPath) as file:
    for line in file:
      valLosses.append(float(line))
  plt.plot(np.arange(len(trainLosses)), trainLosses)
  plt.plot(np.arange(len(valLosses)), valLosses)
  plt.legend(['train', 'val'], loc='upper right')
  print(trainLosses)
  print(valLosses)

  
def getEmbeddingsOfTargets(targets, idx2word, wordMap):
  """
    Creates tensor with the embeddings of the words present in the previously packed target reports
    :param targets: packed_sequence tensor with indexes of words in target reports
    :param idx2word: dictionary with index -> word correspondence.
    :param wordMap: dictionary with word -> embedding correspondence
    """
  targetsList = targets.data.tolist()
  word = idx2word[str(targetsList[0])]
  
  # create tensor with embedding of first word
  targetEmbeddings = wordMap[word]
  targetEmbeddings = torch.from_numpy(targetEmbeddings).unsqueeze_(0).to(device)
  
  # concatenate embeddings of all other words
  for index in targetsList[1:]:
    word = idx2word[str(index)]
    embedding = wordMap[word]
    embedding = torch.from_numpy(embedding).unsqueeze_(0).to(device)
    targetEmbeddings = torch.cat((targetEmbeddings, embedding),0)
  return targetEmbeddings


# continuous output (tensor of embed_dim)
def findClosestWord(continuousOutput, embeddings, idx2word):
  """
    Get nearest neighbour to the continuous output provided by the model
    :param continuousOutput: continuous output of size embed_dim
    :param embeddings: embeddings matrix.
    :param idx2word: dictionary with index -> word correspondence.
  """
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  similarity_matrix = cos(continuousOutput.unsqueeze(0), embeddings)

  word_index = torch.argmax(similarity_matrix)
  
  closestNeighbour = idx2word[str(word_index.item())]
  return closestNeighbour


  # predEmbeddings (batch x length Longest caption x embed_dim)
def generatePredictedCaptions(predEmbeddings, decode_lengths, embeddings, idx2word):
  batch_size = predEmbeddings.shape[0]
  captions = []
  for captionNr in range(batch_size):
    caption = findClosestWord(predEmbeddings[captionNr, 0, :].data, embeddings, idx2word) # First word of caption

    for predictedWordEmbedding in predEmbeddings[captionNr,1:decode_lengths[captionNr], :]:
      # print("New word")
      word = findClosestWord(predictedWordEmbedding.data, embeddings, idx2word)
      if word == '.':
        caption += word
      else:
        caption += ' ' + word
    captions.append(caption)
    # print(captions)
  return captions
