import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import re
import sys
sys.path.append('../Utils/')
import pandas as pd

from generalUtilities import *


class ClassXRayDataset(Dataset):
    """MIMIC xray dataset."""


    def __init__(self, imgsDir, labels_csv_path, transform=None):
        """
        Args:
            json_file (string): Path to the json file with captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imgsDir = imgsDir
        self.transform = transform
        self.imgPaths = getFilesInDirectory(imgsDir)
        self.labels = pd.read_csv(labels_csv_path, index_col = 0)


    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):

        imgID = self.imgPaths[idx]
        study = re.findall(r"s\d{8}", imgID)[0][1:]

        image = Image.open(imgID)

        labels =  torch.tensor(self.labels.loc[int(study)].values[1:], dtype=torch.long)


        if self.transform:
            image = self.transform(image)

        return image, labels


