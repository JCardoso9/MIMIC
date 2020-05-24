import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import re
import sys
import pandas as pd
sys.path.append('../Utils/')


from generalUtilities import *


class HierarchicalXRayDataset(Dataset):
    """MIMIC xray dataset."""


    def __init__(self,word2idx_path,   encodedCaptionsJsonFile, captionsLengthsFile, imgsDir, transform=None, maxSize = 372,
                  max_nr_sentences = 22, max_number_words_per_sentence = 79, training=True):
        """
        Args:
            json_file (string): Path to the json file with captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.training = training

        with open(word2idx_path) as json_file:
            self.word2idx = json.load(json_file)

        with open(encodedCaptionsJsonFile) as json_file:
            self.encodedCaptions = json.load(json_file)

        with open(captionsLengthsFile) as json_file:
            self.encodedCaptionsLength = json.load(json_file)


#        if self.training:
#            with open("/home/jcardoso/MIMIC/valReportLengthInSentences.json") as json_file:
#                self.trainReportLengthInSentences = json.load(json_file)

            #with open("/home/jcardoso/MIMIC/valSentencesLengths.json") as json_file:
                #self.trainSentencesLengths = json.load(json_file)

        #self.labels = pd.read_csv(labels_csv_path, index_col = 0)

        self.max_nr_sentences = max_nr_sentences
        self.max_number_words_per_sentence = max_number_words_per_sentence

        self.imgsDir = imgsDir
        self.transform = transform
        self.imgPaths = getFilesInDirectory(imgsDir)
        self.maxSize = maxSize
        self.vocab_size = len(self.word2idx.keys())

    def __len__(self):
        return len(self.encodedCaptions)

    def __getitem__(self, idx):

        imgID = self.imgPaths[idx]
        study = re.findall(r"s\d{8}", imgID)[0][1:]

        image = Image.open(imgID)

        encodedCaptionLength = 0
        encodedCaption = []

        # Start of caption token
        encodedCaptionLength += 1
        encodedCaption.append(self.word2idx['<sos>'])

        # Caption
        encodedCaptionLength += self.encodedCaptionsLength[study]
        encodedCaption = self.encodeCaption(self.encodedCaptions[study], encodedCaption)

        # End of caption token
        encodedCaptionLength += 1
        encodedCaption.append(self.word2idx['<eoc>'])


        #sentences = self.getSentences(encodedCaption)
        #sentences = self.padsentences(sentences)
        #print("Dataset sentences shape",sentences.shape)
        #paddedReport = self.padReport(torch.LongTensor(sentences))

        #print(paddedReport.shape)


        #print(encodedCaption)
        #print("LENGTH:",torch.LongTensor(sentences))
        #print(sentences)



        # Pad captions until all have max_size
        encodedCaption = self.padCaption(torch.LongTensor(encodedCaption), self.maxSize, encodedCaptionLength)

        #labels =  torch.tensor(self.labels.loc[int(study)].values[1:], dtype=torch.long)

        #print("LENGTH:",torch.LongTensor(sentences))
        if self.transform:
            image = self.transform(image)


        #print(self.trainReportLengthInSentences[study])
        #print(self.trainSentencesLengths[study])

        #if self.training:
            #return image,  torch.LongTensor(encodedCaption), encodedCaptionLength, study, self.trainReportLengthInSentences[study], self.trainSentencesLengths[study]

        return image,  torch.LongTensor(encodedCaption), encodedCaptionLength, study

    # Transform caption comprised of list of words into
    # list of ints according to the word -> index hash table
    def encodeCaption(self, caption, encodedCaption):
        for word in caption:
            if word in self.word2idx.keys():
                encodedCaption.append(self.word2idx[word])
            else:
                encodedCaption.append(self.word2idx['<unk>'])
        return encodedCaption

    # Pad caption to max Size
    def padCaption(self, caption, maxSize, encodedCaptionLength):
      nrOfPads = maxSize - encodedCaptionLength
      padIdx = self.word2idx['<pad>']

      padSequence = torch.full([nrOfPads], padIdx,dtype =torch.long)

      return torch.cat((caption, padSequence), 0)

    def padsentences(self, sentences):
      tensors = []
      for sentence in sentences:
          tensors.append(torch.LongTensor(sentence))
      empty = torch.full((self.max_number_words_per_sentence,),self.word2idx['<pad>'],dtype =torch.long)
      tensors.append(empty)
      sentences = pad_sequence(tensors, batch_first=True, padding_value=self.word2idx['<pad>'])
      #print(sentences)
      return sentences[:-1]


    def padReport(self,  sentences):
      nrOfPads = self.max_nr_sentences - sentences.shape[0]
      emptySentences = torch.full((nrOfPads, sentences.shape[1]),self.word2idx['<pad>'],dtype =torch.long)
      return torch.cat((sentences, emptySentences), 0)
      


    def getSentences(self,caption):
        size = len(caption) 
        idx_list = [idx + 1 for idx, val in
            enumerate(caption) if val == self.word2idx['.']] 
  
  
        res = [caption[i: j] for i, j in
            zip([0] + idx_list, idx_list + 
            ([size] if idx_list[-1] != size else []))]
        res[-2].extend(res[-1])
        res = res[:-1]
        return res
