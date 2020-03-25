import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import re

from utils import *


class XRayDataset(Dataset):
    """MIMIC xray dataset."""

    

    def __init__(self,word2idx_path,  encodedCaptionsJsonFile, captionsLengthsFile, imgsDir, transform=None, maxSize = 456):
        """
        Args:
            json_file (string): Path to the json file with captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        with open(word2idx_path) as json_file:
            self.word2idx = json.load(json_file)

        with open(encodedCaptionsJsonFile) as json_file:
            self.encodedCaptions = json.load(json_file)

        with open(captionsLengthsFile) as json_file:
            self.encodedCaptionsLength = json.load(json_file)

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

        encodedCaptionLength += 1
        encodedCaption.append(self.word2idx['<sos>'])
        
        if self.encodedCaptionsLength[study]['findings'] != 0:
          encodedCaptionLength += self.encodedCaptionsLength[study]['findings']
          encodedCaption += self.encodedCaptions[study]['findings']   

        elif self.encodedCaptionsLength[study]['impression'] != 0:
          encodedCaptionLength += self.encodedCaptionsLength[study]['impression']
          encodedCaption += self.encodedCaptions[study]['impression']

     
        elif self.encodedCaptionsLength[study]['last_paragraph'] != 0:
          encodedCaptionLength += self.encodedCaptionsLength[study]['last_paragraph']
          encodedCaption += self.encodedCaptions[study]['last_paragraph']
        
        # else:
        #   print("error: no captions")

        

        # # print("Imp: ", torch.LongTensor([self.encodedCaptionsLength[study]['impression']]))
        # encodedCaptionLength += self.encodedCaptionsLength[study]['impression']
        # encodedCaption += self.encodedCaptions[study]['impression']

      
        # # print("Findi: ", torch.LongTensor([self.encodedCaptionsLength[study]['findings']]))
        # encodedCaptionLength += self.encodedCaptionsLength[study]['findings']
        # encodedCaption += self.encodedCaptions[study]['findings']

      
        # # print("LP: ", torch.LongTensor([self.encodedCaptionsLength[study]['last_paragraph']]))
        # encodedCaptionLength += self.encodedCaptionsLength[study]['last_paragraph']
        # encodedCaption += self.encodedCaptions[study]['last_paragraph']

        encodedCaptionLength += 1
        encodedCaption.append(self.word2idx['<eoc>'])

        #filteredOOV = [x if x < vocab_size else vocab_size - 1 for x in encodedCaption]
        
        #print(encodedCaption)
        encodedCaption = self.padCaption(torch.LongTensor(encodedCaption), self.maxSize, encodedCaptionLength)


        
        if self.transform:
            image = self.transform(image)

        return image,  torch.LongTensor(encodedCaption), encodedCaptionLength

    # Awful code
    def filterOOV(self, encodedCaption):
      filtered = []
      for index in encodedCaption:
        if index == 17941:
          filtered.append(self.word2idx['.'])
        elif index >= 5368:
          filtered.append(self.word2idx['<unk>'])
        else:
          filtered.append(index)
      return filtered

    def padCaption(self, caption, maxSize, encodedCaptionLength):      
      nrOfPads = maxSize - encodedCaptionLength
      padIdx = self.word2idx['<pad>']

      padSequence = torch.full([nrOfPads], padIdx,dtype =torch.long)

      return torch.cat((caption, padSequence), 0)
    


    
