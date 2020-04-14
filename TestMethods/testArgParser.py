import argparse

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('--runType', type=str, default='',
                        help='type of run: Training / Testing')

    parser.add_argument('--model', type=str, default='Continuous',
                        help='type of model: Softmax / Continuous')

    parser.add_argument('--model_name', type=str, default='testModel',
                        help='name of the model to be stored in results')

    parser.add_argument('--checkpoint', type=str, default=None , metavar='N',
                        help='Path to the model\'s checkpoint (No checkpoint: empty string)')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='define batch size to train the model')

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

    parser.add_argument('--embeddingsPath', type=str, default='',
                        help='path to the embeddings dictionary')

    parser.add_argument('--loss', type=str, default='CosineSim',
                        help='loss to be used')

    parser.add_argument('--normalizeEmb', type=bool, default=False,
                        help='normalize embeddings?')

    parser.add_argument('--attention_dim', type=int,
                        default=512, help='define attention dim')

    parser.add_argument('--decoder_dim', type=int,
                        default=512, help='define decoder dim')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='define dropout probability')

    parser.add_argument('--print_freq', type=int,
                        default=5, help='define print freq of loss')

#    opts, _ = parser.parse_known_args()

 #   args = parser.parse_args()

    args, unknown = parser.parse_known_args()

    return args
