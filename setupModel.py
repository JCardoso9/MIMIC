
def setupModel(args):


  word_map, embeddings, vocab_size, embed_dim = loadEmbeddingsFromDisk(args.embeddingsPath, args.normalizeEmb)


  if (args.model == 'Continuous'):
    decoder = ContinuousOutputDecoderWithAttention(attention_dim=args.attention_dim,
                                    embed_dim=embed_dim
                                    decoder_dim=args.decoder_dim,
                                    vocab_size=vocab_size,
                                    dropout=args.dropout)

  else:
    decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                    embed_dim=embed_dim,
                                    decoder_dim=args.decoder_dim,
                                    vocab_size=vocab_size,
                                    dropout=args.dropout)

  decoder.load_pretrained_embeddings(embeddings)  
  encoder = Encoder()

  # Load trained model if checkpoint exists
  if (args.checkpoint not None):
    modelInfo = torch.load(args.checkpoint)
    decoder.load_state_dict(modelInfo['decoder'])
    encoder.load_state_dict(modelInfo['encoder'])

  # Move to GPU, if available
  decoder = decoder.to(device)
  encoder = encoder.to(device)

  if (args.runType == "Testing"):
    decoder.eval()
    encoder.eval()

  if (args.runType == "Training"):
    #"create optimizer"
   # if checkpoint not non
        #load state dict


  if (args.loss == 'Softmax'):
    criterion = nn.CrossEntropyLoss().to(device)

  elif (args.loss == 'CosineSimilarity'):
    criterion = nn.CosineEmbeddingsLoss().to(device)

  elif (args.loss == 'TripleMarginLoss'):
    #criterion = tripletmarginloss

  return encoder, decoder, criterion, embeddings, word_map





