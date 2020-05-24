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
      super(SmoothL1LossWordAndSentence, self).__init__()
      # Loss function
      self.criterion = nn.SmoothL1Loss().to(device)
      self.beta = beta

    def forward(self, targets, preds, decode_lengths):

      word_loss = 0.
      sentence_loss = 0.
      batch_size = targets.shape[0]
      #unpadded_targets = targets[:,:decode_lengths,:]
      #unpadded_preds = preds[:,:decode_lengths,:]
      for sentence_idx in range(batch_size):
          word_loss += self.criterion(preds[sentence_idx, :decode_lengths[sentence_idx],:], targets[sentence_idx, :decode_lengths[sentence_idx],:])
          

          #print(torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim= 0).shape)
          sentence_loss += self.criterion(torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim= 0),
                                          torch.mean(targets[sentence_idx, :decode_lengths[sentence_idx],:], dim=0))


      #print((1 - self.beta) * (word_loss / batch_size))
      #print(self.beta * (sentence_loss / batch_size))
      return  (word_loss / batch_size) + 0 * (sentence_loss / batch_size)
      #return (1 - self.beta) * (word_loss / batch_size) +   0 *self.beta * (sentence_loss / batch_size)
#      return 1


class SyntheticTripletLoss(nn.Module):
    """
    Triplet margin Loss using syntethically created negative examples

    """

    def __init__(self, margin=0.5, mode='Ortho'):
      super(SyntheticTripletLoss, self).__init__()
      self.margin = margin
      self.mode = mode


    def forward(self, targets, preds, decode_lengths):
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

      #marginT = torch.ones(targets.shape[0]).to(device)
      #marginT = marginT.new_full((targets.shape[0],), self.margin, dtype=torch.float, device=device, requires_grad=False)

      lossPerWord = self.margin + simPredToNegSample - simPredToTarget
      lossPerWord[lossPerWord<0] = 0 # Max(0, loss)

      loss = torch.mean(lossPerWord).item()

      #If using L2, can just apply pytorch function?
      # triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
      #output = triplet_loss(marginT, positive, negSamples)

      return loss



class SyntheticTripletLossSmoothL1Sentence(nn.Module):
    """
    Triplet margin Loss using syntethically created negative examples

    """

    def __init__(self, margin=0.5, mode='Ortho'):
      super(SyntheticTripletLoss, self).__init__()
      self.margin = margin
      self.mode = mode
      self.criterion =  nn.SmoothL1Loss(reduction='none').to(device)

    def forward(self, targets, preds, decode_lengths):
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

      word_loss = 0.
      sentence_loss = 0.
      batch_size = targets.shape[0]
      for sentence_idx in range(batch_size):
          
          distPredToTarget = self.criterion(preds[sentence_idx, :decode_lengths[sentence_idx],:], targets[sentence_idx, :decode_lengths[sentence_idx],:])
          distPredToNegSample = self.criterion(preds[sentence_idx, :decode_lengths[sentence_idx],:], negSamples[sentence_idx, :decode_lengths[sentence_idx],:])

          #marginT = torch.ones(targets.shape[0]).to(device)
          #marginT = marginT.new_full((targets.shape[0],), self.margin, dtype=torch.float, device=device, requires_grad=False)


          word_loss += torch.clamp(self.margin + distPredToNegSample - distPredToNegSample, min=0).mean()


          #print(torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim= 0).shape)
          sentenceDistPredToTarget = self.criterion(torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim= 0),
                                          torch.mean(targets[sentence_idx, :decode_lengths[sentence_idx],:], dim=0))

          sentenceDistPredToNeg = self.criterion(torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim= 0),
                                          torch.mean(negSamples[sentence_idx, :decode_lengths[sentence_idx],:], dim=0))


          sentence_loss += torch.clamp(self.margin + sentenceDistPredToTarget - sentencesDistPredToNeg, min=0).mean()

      #print((1 - self.beta) * (word_loss / batch_size))
      #print(self.beta * (sentence_loss / batch_size))
      return (1 - self.beta) * (word_loss / batch_size) +   self.beta * (sentence_loss / batch_size)



class SmoothL1LossWordAndSentenceAndImage(nn.Module):
    '''
      Uses pytorch's cosine embedding loss. Class created to abstract need of
      using a y vector.
    '''

    def __init__(self, beta=0.5):
      super(SmoothL1LossWordAndSentence, self).__init__()
      # Loss function
      self.criterion = nn.SmoothL1Loss().to(device)
      self.beta = beta

    def forward(self, targets, preds, decode_lengths):

      word_loss = 0.
      sentence_loss = 0.
      image_embedding_loss = 0.

      batch_size = targets[0].shape[0]
      #unpadded_targets = targets[:,:decode_lengths,:]
      #unpadded_preds = preds[:,:decode_lengths,:]
      for sentence_idx in range(batch_size):
          word_loss += self.criterion(preds[sentence_idx, :decode_lengths[sentence_idx],:], targets[0][sentence_idx, :decode_lengths[sentence_idx],:])


          #print(torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim= 0).shape)
          pred_sentence_mean = torch.mean(preds[sentence_idx, :decode_lengths[sentence_idx],:], dim=0)

          sentence_loss += self.criterion(pred_sentence_mean,
                                          torch.mean(targets[0][sentence_idx, :decode_lengths[sentence_idx],:], dim=0))

          image_embedding_loss += self.criterion(pred_sentence_mean, torch.nn.functional.normalize(targets[1][sentence_idx], p=2))

      #print((1 - self.beta) * (word_loss / batch_size))
      #print(self.beta * (sentence_loss / batch_size))
      return  (word_loss / batch_size) + (sentence_loss / batch_size) + image_embedding_loss /batch_size
