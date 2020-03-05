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

    def __len__(self):
        return len(self.encodedCaptions)

    def __getitem__(self, idx):
        
        imgID = self.imgPaths[idx]
        study = re.findall(r"s\d{8}", imgID)[0][1:]
        
        image = Image.open(imgID)
   

        # if self.encodedCaptionsLength[study]['impression'] != 0:
        #   print("Imp: ", torch.LongTensor([self.encodedCaptionsLength[study]['impression']]))
        #   encodedCaptionLength = torch.LongTensor([self.encodedCaptionsLength[study]['impression']])
        #   encodedCaption = torch.LongTensor(self.encodedCaptions[study]['impression']) 

        # elif self.encodedCaptionsLength[study]['findings'] != 0:
        #   print("Imp: ", torch.LongTensor([self.encodedCaptionsLength[study]['findings']]))
        #   encodedCaptionLength = torch.LongTensor([self.encodedCaptionsLength[study]['findings']])
        #   encodedCaption = torch.LongTensor(self.encodedCaptions[study]['findings'])  

        # elif self.encodedCaptionsLength[study]['last_paragprah'] != 0:
        #   print("Imp: ", torch.LongTensor([self.encodedCaptionsLength[study]['last_paragraph']]))
        #   encodedCaptionLength = torch.LongTensor([self.encodedCaptionsLength[study]['last_paragraph']])
        #   encodedCaption = torch.LongTensor(self.encodedCaptions[study]['last_paragraph'])
        
        # else:
        #   print("error: no captions")

        encodedCaptionLength = 0
        encodedCaption = []

        encodedCaptionLength += 1
        encodedCaption.append(self.word2idx['<sos>'])

        # print("Imp: ", torch.LongTensor([self.encodedCaptionsLength[study]['impression']]))
        encodedCaptionLength += self.encodedCaptionsLength[study]['impression']
        encodedCaption += self.encodedCaptions[study]['impression']

      
        # print("Findi: ", torch.LongTensor([self.encodedCaptionsLength[study]['findings']]))
        encodedCaptionLength += self.encodedCaptionsLength[study]['findings']
        encodedCaption += self.encodedCaptions[study]['findings']

      
        # print("LP: ", torch.LongTensor([self.encodedCaptionsLength[study]['last_paragraph']]))
        encodedCaptionLength += self.encodedCaptionsLength[study]['last_paragraph']
        encodedCaption += self.encodedCaptions[study]['last_paragraph']

        encodedCaptionLength += 1
        encodedCaption.append(self.word2idx['<eoc>'])
        
        #print(encodedCaption)
        encodedCaption = self.padCaption(torch.LongTensor(encodedCaption), self.maxSize, encodedCaptionLength)

        # if encodedCaptionLength > 350:
        #   print("From encoded caption length")
        #   print(encodedCaptionLength)
        #   print(study)

        

        # if torch.LongTensor(encodedCaption).shape[0] > 350:
        #   print("From tensor length")
        #   print("Impressions Size: ", self.encodedCaptionsLength[study]['impression'])
        #   print("Findings Size: ", self.encodedCaptionsLength[study]['findings'])
        #   print("Last paragraph Size: ", self.encodedCaptionsLength[study]['last_paragraph'])
        #   print("Actual Impressions Size: ", len(self.encodedCaptions[study]['impression']))
        #   print("Actual Findings Size: ", len(self.encodedCaptions[study]['findings']))
        #   print("Actual Last paragraph Size: ", len(self.encodedCaptions[study]['last_paragraph']))
        #   print(torch.LongTensor(encodedCaption).shape[0])
        #   print(study)

        
        if self.transform:
            image = self.transform(image)

        return image,  torch.LongTensor(encodedCaption), encodedCaptionLength

    def padCaption(self, caption, maxSize, encodedCaptionLength):
      # print("MaxSize:", maxSize)
      # print("Encoded caption length: ", encodedCaptionLength)
      
      nrOfPads = maxSize - encodedCaptionLength
      # print("nrOfPads: ", nrOfPads)
      padIdx = self.word2idx['<pad>']

      padSequence = torch.full([nrOfPads], padIdx,dtype =torch.long)

      return torch.cat((caption, padSequence), 0)
    


    