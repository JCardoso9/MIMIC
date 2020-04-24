
import sys
sys.path.append('../Utils/')

from setupEnvironment import *
from argParser import *
from TrainingEnvironment import *
from generalUtilities import *

import torch
import torch.backends.cudnn as cudnn

import time
from datetime import datetime

from nlgeval import NLGEval


def main():
  """
    Performs testing on the trained model.
    :param modelInfoPath: Path to the model saved during the training process
  """

  argParser = get_args()

  print(argParser)

  # Create NlG metrics evaluator
  nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

  encoder, decoder, criterion, embeddings, word_map, _, _ , idx2word, word2idx= setupModel(argParser)
  embeddings = embeddings.to(device)

  # Create data loaders
  testLoader, _ = setupDataLoaders(argParser)

  # Load word <-> embeddings matrix index correspondence dictionaries

  references, hypotheses = test(argParser, testLoader=testLoader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                idx2word=idx2word,
                                word_map=word_map,
                                embeddings=embeddings)


  metrics_dict = nlgeval.compute_metrics(references, hypotheses)

  with open('../Experiments/' + argParser.model_name + "/TestResults.txt", "w+") as file:
    for metric in metrics_dict:
      file.write(metric + ":" + str(metrics_dict[metric]) + "\n")



if __name__ == "__main__":
  main()
