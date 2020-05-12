
import sys
sys.path.append('../Dataset/')
sys.path.append('../Models/')
sys.path.append('../Utils/')

from torchvision.models import densenet161, resnet50, vgg19
from XRayDataset import *
from generalUtilities import *
import faiss
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



torch.manual_seed(2809)
torch.backends.cudnn.deterministic = True

# Create NlG metrics evaluator
nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])


class Encoder(nn.Module):
    def __init__(self, network='resnet50'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet50':
            self.net = resnet50(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.dim = 2048

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x

encoder = Encoder()
encoder = encoder.to(device)
encoder.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


batch_size = 4


trainLoader = DataLoader(XRayDataset('/home/jcardoso/MIMIC/word2idxF.json', '/home/jcardoso/MIMIC/encodedTrainCaptionsF.json',
      '/home/jcardoso/MIMIC/encodedTrainCaptionsLengthsF.json', '/home/jcardoso/MIMIC/TrainF',  transform), batch_size=batch_size, shuffle=False)

testLoader = DataLoader(XRayDataset('/home/jcardoso/MIMIC/word2idxF.json', '/home/jcardoso/MIMIC/encodedTestCaptionsF.json',
      '/home/jcardoso/MIMIC/encodedTestCaptionsLengthsF.json', '/home/jcardoso/MIMIC/TestF',  transform), batch_size=batch_size, shuffle=False)


with open('/home/jcardoso/MIMIC/idx2wordF.json') as fp:
    idx2word = json.load(fp)

with open('/home/jcardoso/MIMIC/word2idxF.json') as fp:
    word2idx = json.load(fp)


d = 2048
index = faiss.IndexFlatIP(d)

train_captions = {}

#with open('train_captionsNN.pkl', 'rb') as file:
    #train_captions =  pickle.load(file)





def generate_train_images_matrix():

    count = 0
    rounds = 0
    print("Nr images: ", len(trainLoader))
    for i, (image, caps, caplens) in enumerate(
            tqdm(trainLoader, desc="Creating image matrix: ")):
        image = image.to(device)
        encoder_out = encoder(image)
        encoder_out = encoder_out.view(batch_size,-1)

        encoder_out = torch.nn.functional.normalize(encoder_out, p=2, dim=1)
        index.add(encoder_out.to("cpu").detach().numpy())
        encodedCaption = [w for w in caps[0].tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
#        print(decodeCaption(encodedCaption, idx2word))

#        print("ººººººº")
        for caption in caps:
#            print("-----------")
            encodedCaption = [w for w in caption.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
#            print(decodeCaption(encodedCaption, idx2word))
            train_captions[count] = decodeCaption(encodedCaption, idx2word)
            count += 1
#        print("~~~~~~~~~~")
        #print(train_captions)
#        print("Saved:",train_captions[0])
        #break
#        print("final:",train_captions)
        rounds += 1
        if rounds > 38000:
            break

    #with open('train_captionsNN.pkl', 'wb') as file:
        #pickle.dump(train_captions, file)
    #return index

k=1

def calculate_NN():
    references = [[]]
    hypotheses = []
    for i, (image, caps, caplens) in enumerate(
            tqdm(testLoader, desc="Calculating NN: ")):
        image = image.to(device)
        encoder_out = encoder(image)
        encoder_out = encoder_out.view(batch_size,-1)
        encoder_out = torch.nn.functional.normalize(encoder_out, p=2, dim=1)
        D, I = index.search(encoder_out.to("cpu").detach().numpy(), k)


        for caption in caps:
            encodedCaption = [w for w in caption.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
            references[0].append(decodeCaption(encodedCaption, idx2word))
        #print("\nD:",D)
        #print("I",I)
        #print(I.shape)
        #print(I[0][0])

        for i in range(batch_size):
            hypotheses.append(train_captions[I[i][0]])

        #break

    return references, hypotheses




generate_train_images_matrix()

references, hypotheses = calculate_NN()

print(len(references[0]))

print("HYPS:", len(hypotheses))

metrics_dict = nlgeval.compute_metrics(references, hypotheses)
print(metrics_dict)

with open("AllRefs.txt", 'w+') as file:
    for reference in references[0]:
        file.write(reference.strip() + '\n')
with open("AllPreds.txt", 'w+') as file:
    for hypothesis in hypotheses:
        file.write(hypothesis.strip() + '\n')
#with open("nearestN_TestResults.txt", "w+") as file:
   #for metric in metrics_dict:
      #file.write(metric + ":" + str(metrics_dict[metric]) + "\n")








