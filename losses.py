import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class CosineEmbedLoss(nn.Module):
    '''
      Uses pytorch's cosine embedding loss. Class created to abstract need of
      using a y vector.
    '''

    def __init__(self):
      super(CosineEmbedLoss, self).__init__()
      # Loss function
      self.criterion = nn.CosineEmbeddingLoss().to(device)


    def forward(self, targets, preds, decode_lengths):
      preds = pack_padded_sequence(preds, decode_lengths, batch_first=True)
      targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
      y = torch.ones(targets.data.shape[0]).to(device)
      loss = self.criterion(preds.data, targets.data,y)
      return loss



class SmoothL1LossWord(nn.Module):
    '''
      Uses pytorch's cosine embedding loss. Class created to abstract need of
      using a y vector.
    '''

    def __init__(self):
      super(SmoothL1LossWord, self).__init__()
      # Loss function
      self.criterion = nn.SmoothL1Loss().to(device)


    def forward(self, targets, preds, decode_lengths):
      preds = pack_padded_sequence(preds, decode_lengths, batch_first=True)
      targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)
      loss = self.criterion(preds.data, targets.data)
      return loss



class SmoothL1LossWordAndSentence(nn.Module):
    '''
      Uses pytorch's cosine embedding loss. Class created to abstract need of
      using a y vector.
    '''

    def __init__(self, beta=0.5):
      super(SmoothL1LossWord, self).__init__()
      # Loss function
      self.criterion = nn.SmoothL1Loss().to(device)
      self.beta = beta

    def forward(self, targets, preds, decode_lengths):

      word_loss = 0.
      sentence_loss = 0.
      batch_size = unpadded.shape[0]
      unpadded_targets = targets[:,:decode_lengths,:]
      unpadded_preds = preds[:,:decode_lengths,:]
#      for sentence_idx in range(batch_size):
#          word_loss += self.criterion(unpadded_preds[sentence_idx], unpadded_targets[sentence_idx])

#          sentence_loss += self.criterion(torch.mean(unpadded_preds[sentence_idx], dim=0) ,torch.mean(unpadded_targets[sentence_idx],dim=0)



 #     return 1 - self.beta * (word_loss / batch_size) +  self.beta * (sentence_loss / batch_size)
      return 1


class SmoothL1LossWordAndSentenceAndImgRetrieval(nn.Module):
    '''
      Uses pytorch's cosine embedding loss. Class created to abstract need of
      using a y vector.
    '''

    def __init__(self, beta=0.3, img_embed_weight = 0.3):
      super(SmoothL1LossWord, self).__init__()
      # Loss function
      self.criterion = nn.SmoothL1Loss().to(device)
      self.beta = beta
      self.img_embed_weight = img_embed_weight

    def forward(self, targets, preds, decode_lengths):

      img_embedding = targets[1]
      targets = targets[0]

      word_loss = 0.
      sentence_loss = 0.
      img_embed_loss = 0.

      batch_size = unpadded.shape[0]
      unpadded_targets = targets[:,:decode_lengths,:]
      unpadded_preds = preds[:,:decode_lengths,:]

#      for sentence_idx in range(batch_size):
#          word_loss += self.criterion(unpadded_preds[sentence_idx], unpadded_targets[sentence_idx])

#          sentence_loss += self.criterion(torch.mean(unpadded_preds[sentence_idx], dim=0) ,torch.mean(unpadded_targets[sentence_idx],dim=0)
#           img_embed_loss += self.criterion(torch.mean(unpadded_preds[sentence_idx], dim=0), img_embedding)


 #     return (1. - self.beta - self.img_embed_weight) * (word_loss / batch_size) +  self.beta * (sentence_loss / batch_size) +  self.img_embed_weight * (img_embed_loss / batch_size)
      return 1





class SyntheticTripletLoss(nn.Module):
    """
    Triplet margin Loss using syntethically created negative examples

    """

    def __init__(self, margin=0.5, mode='Ortho'):
      super(SyntheticTripletLoss, self).__init__()
      self.margin = margin
      self.mode = mode


    def forward(self, targets, preds):
      '''
      Can create synthetic examples based on an orthogonal or subtraction basis
      :param targets: target embeddings, a tensor of dimensions (sum length captions in batch, embedding_dim)
      :param preds: predicted embeddings, a tensor of dimensions (sum length captions in batch, embedding_dim)
      :return: mean loss across all words
      '''

      # Create a negative example by projecting the predicted embedding to the target embedding and use that
      # to find component of the pred embedding orthogonal to the target embedding
      # û− (û.T dotprod u)u   ---- û: pred embedding, u: target embedding
      if self.mode == 'Ortho':
        negSamples = preds - torch.mm(preds, targets.T).diag().unsqueeze(1).expand(targets.shape[0], targets.shape[1]) * targets

      # Create negative sample by subtraction
      # û - u  ----û: pred embedding, u: target embedding
      elif self.mode == 'Diff':
        negSamples = preds - targets

      negSamples = torch.nn.functional.normalize(negSamples, p=2, dim=1)
      # embeddings normalized -> cosine sim = dot product
      simPredToTarget = torch.mm(preds, targets.T).diag()
      simPredToNegSample = torch.mm(preds, negSamples.T).diag()

      marginT = torch.ones(targets.shape[0]).to(device)
      marginT = marginT.new_full((targets.shape[0],), self.margin, dtype=torch.float, device=device, requires_grad=False)

      lossPerWord = marginT + simPredToNegSample - simPredToTarget
      lossPerWord[lossPerWord<0] = 0 # Max(0, loss)

      loss = torch.mean(lossPerWord).item()

      #If using L2, can just apply pytorch function?
      # triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
      #output = triplet_loss(marginT, positive, negSamples)

      return loss
