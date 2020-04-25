import torch


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
    decodedCaption = idx2word[str(caption[0])]

    for index in caption[1:-1]:
      word = idx2word[str(index)]
      if word == '.':
          decodedCaption += word
      else:
          decodedCaption += ' ' + word

    return decodedCaption


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
    return optimizer.param_groups[0]['lr']

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



