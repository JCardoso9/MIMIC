import argparse

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--runType', type=str, default='',
                        help='type of run: Training / Testing')

    parser.add_argument('--model', type=str, default='Continuous',
                        help='type of model: Softmax / Continuous')

    parser.add_argument('--model_name', type=str, default='testModel',
                        help='name of the model to be stored in results')

    parser.add_argument('--encoder_name', type=str, default='',
                        help='name of the pre trained encoder model to be used')


    parser.add_argument('--checkpoint', type=str, default=None , metavar='N',
                        help='Path to the model\'s checkpoint (No checkpoint: empty string)')

    parser.add_argument('--use_img_embedding', type=bool, default=False,
                        help='use image embedding in loss computation?')

    parser.add_argument('--use_classifier_encoder', type=bool, default=False,
                        help='use a classifier pre trained in Xray classification?')

    parser.add_argument('--classifier_checkpoint', type=str, default=None , metavar='N',
                        help='Path to the model\'s classifier checkpoint (No checkpoint: empty string)')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='define batch size to train the model')

    parser.add_argument('--csv_path', type=str, default="",
                        help='path to the csv file with disease labels')

    parser.add_argument('--word2idxPath', type=str, default="",
                        help='path to the dictionary with word -> embeddings matrix index correspondence')

    parser.add_argument('--idx2wordPath', type=str, default="",
                        help='path to the dictionary with embeddings matrix index -> word correspondence')

    parser.add_argument('--encodedTestCaptionsPath', type=str, default="",
                        help='path to the encoded captions to be used')

    parser.add_argument('--encodedTestCaptionsLengthsPath', type=str, default='',
                        help='path to the encoded captions lengths to be used')
 
    parser.add_argument('--testImgsPath', type=str, default='',
                        help='path to the images to be used')

    parser.add_argument('--encodedTrainCaptionsPath', type=str, default="",
                        help='path to the encoded captions to be used')

    parser.add_argument('--encodedTrainCaptionsLengthsPath', type=str, default='',
                        help='path to the encoded captions lengths to be used')
 
    parser.add_argument('--trainImgsPath', type=str, default='',
                        help='path to the images to be used')

    parser.add_argument('--encodedValCaptionsPath', type=str, default="",
                        help='path to the encoded captions to be used')

    parser.add_argument('--encodedValCaptionsLengthsPath', type=str, default='',
                        help='path to the encoded captions lengths to be used')
 
    parser.add_argument('--valImgsPath', type=str, default='',
                        help='path to the images to be used')

    parser.add_argument('--embeddingsPath', type=str, default='',
                        help='path to the embeddings dictionary')

    parser.add_argument('--loss', type=str, default='CosineSim',
                        help='loss to be used')

    parser.add_argument('--triplet_loss_margin', type=float, default=0.5,
                        help='margin used in triplet margin loss computation')

    parser.add_argument('--triplet_loss_mode', type=str, default='Ortho',
                        help='Negative sampling mode: Ortho, Dif, MINS.')

    parser.add_argument('--normalizeEmb', type=bool, default=False,
                        help='normalize embeddings?')

    parser.add_argument('--attention_dim', type=int,
                        default=512, help='define attention dim')

    parser.add_argument('--decoder_dim', type=int,
                        default=512, help='define decoder dim')

    parser.add_argument('--hidden_dim', type=int,
                        default=512, help='define decoder dim')


    parser.add_argument('--dropout', type=float, default=0.5,
                        help='define dropout probability')

    parser.add_argument('--print_freq', type=int,
                        default=5, help='define print freq of loss')

    parser.add_argument('--fine_tune_embeddings', type=bool, default=False,
                        help='fine tune embeddings?')

    parser.add_argument('--fine_tune_encoder', type=bool, default=False,
                        help='fine tune encoder?')


    parser.add_argument('--encoder_lr', type=float, default=1e-3,
                        help='encoder learning rate')

    parser.add_argument('--decoder_lr', type=float, default=1e-3,
                        help='decoder learning rate')


    parser.add_argument('--epochs', type=int, default=20,
                        help='max number of epochs during training')

    parser.add_argument('--lr_threshold', type=float, default=1e-5,
                        help='threshold for decaying learning rate')

    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='value to clip gradients by')

    parser.add_argument('--alpha_c', type=float, default=1.,
                        help='regularization parameter for doubly stochastic attention')

    parser.add_argument('--use_tf_as_input', type=int, default=1,
                        help='Use teacher forcing during training?')

    parser.add_argument('--early_stop_epoch_threshold', type=int, default=5,
                        help='Number of epochs without improvement which leads to early stop')

    parser.add_argument('--lr_decay', type=float, default=.5,
                        help='learning rate decay used by scheduler')

    parser.add_argument('--lr_decay_epochs', type=int, default=5,
                        help='Nr epochs before scheduler lowers learning rate')

    parser.add_argument('--use_scheduled_sampling', type=bool, default=False,
                        help='Use scheduled sampling during training?')

    parser.add_argument('--initial_scheduled_sampling_prob', type=float, default=.0,
                        help='initial probability to sample from previous decoder output')

    parser.add_argument('--scheduled_sampling_decay_epochs', type=int, default=5,
                        help='Nr epochs before probability to sample from previous decoder output is lowered')


    parser.add_argument('--rate_change_scheduled_sampling_prob', type=float, default=.05,
                        help='increment to the probability of sampling from previous decoder output')


    parser.add_argument('--use_custom_tf', type=bool, default=False,
                        help='use custom tf method in continuous model?')


    parser.add_argument('--use_mogrifier', type=bool, default=False,
                        help='use mogrifier lstm?')

#    opts, _ = parser.parse_known_args()

 #   args = parser.parse_args()

    args, unknown = parser.parse_known_args()

    return args
