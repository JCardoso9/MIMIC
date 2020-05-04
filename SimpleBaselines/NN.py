
import sys
sys.path.append('../Dataset/')
sys.path.append('../Models/')
sys.path.append('../Utils/')


from torchvision.models import densenet161, resnet152, vgg19
from XRayDataset import *
from generalUtilities import *

from nlgeval import NLGEval

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import json
from torchvision import transforms
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create NlG metrics evaluator
nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])


#encoder = Encoder()

encoder = resnet152(pretrained=True)
encoder = nn.Sequential(*list(encoder.children())[:-1])

encoder = encoder.to(device)
encoder.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


batch_size = 1

trainLoader = DataLoader(XRayDataset('/home/jcardoso/MIMIC/word2idxF.json', '/home/jcardoso/MIMIC/encodedTrainCaptionsF.json',
      '/home/jcardoso/MIMIC/encodedTrainCaptionsLengthsF.json', '/home/jcardoso/MIMIC/TrainF', transform), batch_size=batch_size, shuffle=True)


valLoader = DataLoader(XRayDataset('/home/jcardoso/MIMIC/word2idxF.json', '/home/jcardoso/MIMIC/encodedTestCaptionsF.json',
      '/home/jcardoso/MIMIC/encodedTestCaptionsLengthsF.json', '/home/jcardoso/MIMIC/TestF',  transform), batch_size=batch_size, shuffle=True)


with open('/home/jcardoso/MIMIC/idx2wordF.json') as fp:
    idx2word = json.load(fp)

with open('/home/jcardoso/MIMIC/word2idxF.json') as fp:
    word2idx = json.load(fp)


train_images_matrix = torch.zeros(len(trainLoader), 2048)
train_captions = {}


def generate_train_images_matrix():
    count = 0

    print("Nr images: ", len(trainLoader))
    for i, (image, caps, caplens) in enumerate(
            tqdm(trainLoader, desc="Creating image matrix: ")):
        image = image.to(device)
        encoder_out = encoder(image)

        train_images_matrix[i: i+batch_size, :] = encoder_out
        print(train_images_matrix.shape)
        for caption in caps:
            encodedCaption = [w for w in caption.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
            train_captions[i] = decodeCaption(encodedCaption, idx2word)
            count += 1
        print(train_captions)
        break

    #with open('train_images_matrix.obj', 'w') as file:
    #    pickle.dump(train_images_matrix, file)
    #with  open('train_captions.obj', 'w') as file:
    #    pickle.dump(train_captions, file)



def calculate_NN():
    references = [[]]
    hypotheses = []
    for i, (image, caps, caplens) in enumerate(
            tqdm(testLoader, desc="Creating image matrix: ")):
        image = image.to(device)
        encoder_out = encoder(image)


        for caption in caps:
            encodedCaption = [w for w in caption.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
            references[0].append(decodeCaption(encodedCaption, idx2word))

        #similarity_matrix = torch.mm(img?, train_images_matrix.T)
        nn_report_index = torch.argmax(similarity_matrix, dim=1)

        hypotheses.append(train_captions[nn_report_index])

    return references, hypotheses


#with open('train_images_matrix.obj', 'r') as file:
    #train_image_matrix =  pickle.load(file)

#with open('train_captions.obj', 'r') as file:
    #train_captions =  pickle.load(file)



generate_train_images_matrix()

#references, hypotheses = calculate_NN()

#metrics_dict = nlgeval.compute_metrics(references, hypotheses)
#print(metrics_dict)

#with open("nearestN_TestResults.txt", "w+") as file:
   #for metric in metrics_dict:
      #file.write(metric + ":" + str(metrics_dict[metric]) + "\n")








