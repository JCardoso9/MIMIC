import torch
import torch.nn as nn

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


    def forward(self, targets, preds):
      y = torch.ones(targets.shape[0]).to(device)
      loss = self.criterion(preds, targets,y)
      return loss




class SyntheticTripletLoss(nn.Module):
    """
    Triplet margin Loss using syntethically created negative examples

    """

    def __init__(self, margin=0.5, mode='Ortho'):
      super(SyntheticTripletLoss, self).__init__()
      self.margin = 0.5
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
