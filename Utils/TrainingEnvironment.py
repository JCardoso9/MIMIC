class TrainingEnvironment:

    """
    Decoder.
    """
    __instance = None
    def __new__(cls, val):
        if TrainingEnvironment.__instance is None:
            TrainingEnvironment.__instance = object.__new__(cls)
        TrainingEnvironment.__instance.val = val
        return TrainingEnvironment.__instance

    def setupTrainingEnvironment(args):
      """
        Setup the training environment by setting the various variables to their
        suitable values.
        :param args: Argument Parser object with definitions set in a specified config file  
      """
      if args.checkpoint is None:
        self.current_lr = args.decoder_lr
        self.start_epoch = modelInfo['epoch'] + 1
        self.epochs_since_improvement = modelInfo['epochs_since_improvement']
        self.best_bleu4 = modelInfo['bleu-4']
        self.best_loss = modelInfo['best_loss']
        self.start_epoch = 0

      else:
        modelInfo = torch.load(args.checkpoint)
        self.current_lr = modelInfo['decoder_optimizer'].param_groups[0]['lr']
        self.start_epoch = modelInfo['epoch'] + 1
        self.epochs_since_improvement = modelInfo['epochs_since_improvement']
        self.best_bleu4 = modelInfo['bleu-4']
        self.best_loss = modelInfo['best_loss']

      self.epochs = args.epochs
      self.lr_threshold = args.lr_threshold
      self.grad_clip = args.grad_clip
      self.alpha_c = args.alpha_c
      self.print_freq = args.print_freq

      self.workers = args.workers




