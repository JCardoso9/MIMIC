import torch
import os

def writeLossToFile(loss, path):
  """
    Write loss of the curren batch to a log file
    :param loss: Average loss of the current batch.
    :param path: path to the log file
  """
  with open(path, 'a+') as file:
    file.write(str(loss) + '\n')



def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def getFilesInDirectory(path):
  """
    Get the path of all files in this directory, recursively
    the embeddings dimension
    :param path: Root path. 
  """
  files = []
  for r, d, f in os.walk(path):
      for file in f:
          if '.jpg' in file:
              files.append(os.path.join(r, file))
  return files


def decodeCaption(caption, idx2word):
    if len(caption) == 0:
        return 'empty'

    decodedCaption = idx2word[str(caption[0])]

    for index in caption[1:]:
      word = idx2word[str(index)]
      if word == '<eoc>':
          break
      if word == '.':
          decodedCaption += word
      else:
          decodedCaption += ' ' + word

    return decodedCaption

def decodeReference(caption, idx2word, word2idx):
    caption = [w for w in caption if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
    decodedCaption = idx2word[str(caption[0])]

    for index in caption[1:-1]:
      word = idx2word[str(index)]
      if word == '.':
          decodedCaption += word
      else:
          decodedCaption += ' ' + word

    return decodedCaption



def unifyCaption(listCaption):
    unifiedCaption = ''
    for word in listCaption:
      if word == '.':
          unifiedCaption += word
      else:
          unifiedCaption += ' ' + word

    return unifiedCaption


def save_checkpoint(modelName, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, metrics_dict, best_loss):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """

    # dd/mm/YY H:M:S
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'metrics_dict': metrics_dict,
             'best_loss':best_loss}

    filename =  '../Experiments/' + modelName + '/checkpoint.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse c$
    if is_best:
        torch.save(state, '../Experiments/' + modelName + '/BEST.pth.tar')



def save_references_and_predictions(references, predictions, modelName, mode):
    refs_path = "../Experiments/" + modelName +"/" + mode +"Refs.txt"
    preds_path = "../Experiments/" + modelName +"/" + mode + "Preds.txt"
    with open(refs_path, 'w+') as file:
        for reference in references[0]:
            file.write(reference.strip() + '\n')
    with open(preds_path, 'w+') as file:
        for prediction in predictions:
            file.write(prediction.strip() + '\n')
    return refs_path, preds_path



def reshape_target_labels(targets):
    batch_size = targets.shape[0]
    nr_labels = targets.shape[1]
    reshaped = torch.zeros((batch_size, nr_labels *2), dtype=torch.float)
    for i in range(batch_size):
        #print(i)
        for j in range(nr_labels):
            #print(j)
            if targets[i,j] == -1:
                reshaped[i,j*2] = 0.5
                reshaped[i,j*2+1] = 0.5
            elif targets[i,j] == 1:
                reshaped[i,j*2] = 1
            elif targets[i,j] == 0:
                reshaped[i,j*2+1] = 1
    return reshaped

