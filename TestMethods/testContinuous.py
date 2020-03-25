import argparse, json
from utils import *
from encoder import Encoder
from Attention import *
from ContinuousOutputDecoderWithAttention import *
from XRayDataset import *

import torch
import os
import numpy as np

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

from nlgeval import NLGEval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_freq = 5  # print  stats every __ batches


def test(word_map,embeddings,idx2word, testLoader, encoder, decoder, criterion):
    """
    Performs testing for the pretrained model
    :param word_map: dictionary with word -> embedding correspondence
    :param embeddings: Embeddings matrix 
    :param idx2word: dictionary with index -> word correspondence.
    :param testLoader: loader for test data
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
        for i, (imgs, caps, caplens) in enumerate(testLoader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            predEmbeddings, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            predEmbeddings_copy = predEmbeddings.clone()
            predEmbeddings = pack_padded_sequence(predEmbeddings, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            targets = getEmbeddingsOfTargets(targets, idx2word, word_map)
            y = torch.ones(targets.shape[0]).to(device)
            loss = criterion(predEmbeddings.data, targets,y)

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
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(testLoader), batch_time=batch_time,
                                                                                loss=losses))
                

            # Store references (true captions), and hypothesis (prediction) for each image
            # references = [[ref1, ref2, ...]], hypotheses = [hyp1, hyp2, ...]
        

            temp_refs = []
            caps_sortedList = caps_sorted[:, 1:].tolist()
            for j,refCaption in enumerate(caps_sortedList):
              temp_refs.append(refCaption[:decode_lengths[j]]) 

            for caption in temp_refs:
              references[0].append(decodeCaption(caption, idx2word))

            # Hypotheses
            batch_hypotheses = generatePredictedCaptions(predEmbeddings_copy, decode_lengths, embeddings, idx2word)
            #print(decodedTempRef)

            
            hypotheses.extend(batch_hypotheses)
                    

            assert len(references[0]) == len(hypotheses)


    now = datetime.now()
    day_string = now.strftime("%d_%m_%Y")
    path = 'testLoss' + day_string
    writeLossToFile(losses.avg, path)

    return references, hypotheses



def main(modelInfoPath, modelName):
  """
    Performs testing on the trained model.
    :param modelInfoPath: Path to the model saved during the training process
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Load dictionary with index -> word correspondence
  with open('/home/jcardoso/MIMIC/idx2word.json') as fp:
        idx2word = json.load(fp)

  # Create NlG metrics evaluator
  nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

  #Load embeddings 
  word_map, embeddings, vocab_size, embed_dim = loadEmbeddingsFromDisk('/home/jcardoso/MIMIC/embeddingsMIMIC.pkl')
  embeddings = embeddings.to(device)


  attention_dim = 512  # dimension of attention linear layers
  decoder_dim = 512  # dimension of decoder RNN

  decoder = ContinuousOutputDecoderWithAttention(attention_dim=attention_dim,
                                    embed_dim=embed_dim,
                                    decoder_dim=decoder_dim,
                                    vocab_size=vocab_size,
                                    dropout=0.5)

  decoder.load_pretrained_embeddings(embeddings)  
  encoder = Encoder()

  # Load trained model
  modelInfo = torch.load(modelInfoPath)
  decoder.load_state_dict(modelInfo['decoder'])
  encoder.load_state_dict(modelInfo['encoder'])

  # Move to GPU, if available
  decoder = decoder.to(device)
  encoder = encoder.to(device)

  criterion = nn.CosineEmbeddingLoss().to(device)

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

  # Create MIMIC test dataset loader
  testLoader = DataLoader(
      XRayDataset("/home/jcardoso/MIMIC/word2idx.json","/home/jcardoso/MIMIC/encodedTestCaptions.json",'/home/jcardoso/MIMIC/encodedTestCaptionsLengths.json','/home/jcardoso/MIMIC/Test', transform),
      batch_size=16, shuffle=True)
  
  references, hypotheses = test(word_map, embeddings, idx2word, testLoader=testLoader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
  
  metrics_dict = nlgeval.compute_metrics(references, hypotheses)

  

  with open(modelName + "_TestResults.txt", "w+") as file:
    for metric in metrics_dict:
      file.write(metric + ":" + str(metrics_dict[metric]) + "\n")

  
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Continuous Model')
    parser.add_argument('--checkpoint', type=str, default='', metavar='N',
                        help='Path to the model\'s checkpoint (No checkpoint: empty string)')

    parser.add_argument('--modelName', type=str, default='Continuous', metavar='N',
                        help='Name of the model to write on results file')




    args = parser.parse_args()
    if args.checkpoint:
        main(args.checkpoint, args.modelName)

    else:
        main()
