def getEmbeddingsOfTargets(targets, idx2word, wordMap):
  """
    Creates tensor with the embeddings of the words present in the previously packed target reports
    :param targets: packed_sequence tensor with indexes of words in target reports
    :param idx2word: dictionary with index -> word correspondence.
    :param wordMap: dictionary with word -> embedding correspondence
    """
  targetsList = targets.data.tolist()
  word = idx2word[str(targetsList[0])]

  # create tensor with embedding of first word
  targetEmbeddings = wordMap[word]
  targetEmbeddings = torch.from_numpy(targetEmbeddings).unsqueeze_(0).to(device)
  
  # concatenate embeddings of all other words
  for index in targetsList[1:]:
    word = idx2word[str(index)]
    embedding = wordMap[word]
    embedding = torch.from_numpy(embedding).unsqueeze_(0).to(device)
    targetEmbeddings = torch.cat((targetEmbeddings, embedding),0)
  return targetEmbeddings


# continuous output (tensor of embed_dim)
def findClosestWord(continuousOutput, embeddings, idx2word):
  """
    Get nearest neighbour to the continuous output provided by the model
    :param continuousOutput: continuous output of size embed_dim
    :param embeddings: embeddings matrix.
    :param idx2word: dictionary with index -> word correspondence.
  """
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  similarity_matrix = cos(continuousOutput.unsqueeze(0), embeddings)

  word_index = torch.argmax(similarity_matrix)

  closestNeighbour = idx2word[str(word_index.item())]
  return closestNeighbour, word_index



# predEmbeddings (batch x length Longest caption x embed_dim)
def generatePredictedCaptions(predEmbeddings, decode_lengths, embeddings, idx2word):
  batch_size = predEmbeddings.shape[0]
  captions = []
  for captionNr in range(batch_size):
    caption, _ = findClosestWord(predEmbeddings[captionNr, 0, :].data, embeddings, idx2word) # First wo$

    for predictedWordEmbedding in predEmbeddings[captionNr,1:decode_lengths[captionNr], :]:
      word, _ = findClosestWord(predictedWordEmbedding.data, embeddings, idx2word)
      if word == '.':
        caption += word
      else:
        caption += ' ' + word
    captions.append(caption)

  return captions


