import sys
sys.path.append('../Utils/')
sys.path.append('../')

from setupEnvironment import *
from argParser import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

from nlgeval import NLGEval

MAX_CAPTION_LENGTH = 350

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


def main():
    argParser = get_args()

    print(argParser)

    # Load model
    encoder, decoder, criterion, embeddings, _, _, _, _, idx2word, word2idx =  setupModel(argParser)

    # Create data loaders
    testLoader, _ = setupDataLoaders(argParser)

    # Load word <-> embeddings matrix index correspondence dictionaries
    idx2word, word2idx = loadWordIndexDicts(argParser)

    # Create NlG metrics evaluator
    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])


    references, hypotheses = evaluate_greedy(argParser, encoder, decoder, testLoader, word2idx, idx2word)

    metrics_dict = nlgeval.compute_metrics(references, hypotheses)

    refs_path, preds_path = save_references_and_predictions(references, hypotheses, argParser.model_name, mode="greedy")

    with open('../Experiments/' + argParser.model_name  + "/GreedyTestResults.txt", "w+") as file:
      for metric in metrics_dict:
        file.write(metric + ":" + str(metrics_dict[metric]) + "\n")

def evaluate_greedy(argParser, encoder, decoder, testLoader, word2idx, idx2word):
    """
    Greedy Evaluation

    :param argParser: Argument Parser object with definitions set in a specified config file
    :param encoder: Image Encoder used
    :param decoder: Decoder to generate reports
    :param testLoader: data loader with test data
    :param word2idx: hash table with word -> index correspondence
    :param idx2word: hash table with index -> word correspondence
    :return: hypothesis and references
    """

    encoder, decoder, criterion, embeddings, _, _, _, _, idx2word, word2idx =  setupModel(argParser)

    decoder.eval()
    encoder.eval()

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = [[]]
    hypotheses = list()
    vocab_size = decoder.embedding.weight.shape[0]

    # For each image
    for i, (image, caps, caplens) in enumerate(
            tqdm(testLoader, desc="Evaluating with greedy decoding: ")):

        t = 0

        finished_sequences = []

        # Move to GPU device, if available
        image = image.to(device)  # (batch_size, 3, 224, 224)
        #print(image.shape)
        # Encode
        encoder_out = encoder(image)  # (batch_size, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)
        #print(batch_size)
        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        h, c = decoder.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        nr_ended_sequences = 0

        input = decoder.sos_embedding.expand(batch_size, decoder.embed_dim).to(device) # (batch_size, embed_dim)

        if (argParser.model == "Continuous"):
            predictions = torch.zeros(batch_size, MAX_CAPTION_LENGTH, decoder.embed_dim, dtype=torch.long).to(device)

        elif (argParser.model == "Softmax"):
            predictions = torch.zeros(batch_size, MAX_CAPTION_LENGTH, 1, dtype=torch.long).to(device)

        alphas = torch.zeros(batch_size, MAX_CAPTION_LENGTH , num_pixels).to(device)

        while batch_size > len(finished_sequences):

            awe, alpha = decoder.attention(encoder_out, h)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (batch_size, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([input, awe], dim=1), (h, c))  # (batch_size, decoder_dim)

            if (argParser.model == "Softmax"):
                scores = decoder.fc(h)  # (batch_size, vocab_size)
                scores = F.log_softmax(scores, dim=1)
                pred_word_indexes = torch.argmax(scores, dim=1)
                nr_ended_sequences += pred_word_indexes.tolist().count(word2idx['<eoc>'])
                #print(pred_word_indexes)
                #print("---------------------")
                #sentences_that_produced_eoc = [i for i, e in enumerate(pred_word_indexes.tolist()) if e == word2idx['<eoc>']]
                #print(sentences_that_produced_eoc)
                #finished_sequences.extend([i for i in sentences_that_produced_eoc if i not in finished_sequences])
                #print(finished_sequences)
                #nr_ended_sequences += pred_word_indexes.tolist().count(word2idx['<eoc>'])
                #print(nr_ended_sequences)


            elif (argParser.model == "Continuous"):
                preds = decoder.fc(h) # (batch_size, embed_dim)
                preds =  torch.nn.functional.normalize(preds, p=2, dim=1)
                similarity_matrix = torch.mm(preds, decoder.embedding.weight.T)
                pred_word_indexes = torch.argmax(similarity_matrix, dim=1)
                nr_ended_sequences += pred_word_indexes.tolist().count(word2idx['<eoc>'])

            #print(pred_word_indexes.shape)
            predictions[:, t, :] = pred_word_indexes
            #print(predictions[0,:,:])
            #print("------------------------")
            alphas[:, t, :] = alpha

            input = decoder.embedding(pred_word_indexes)
            t +=1

            if t > MAX_CAPTION_LENGTH - 1:
                #print("Reached maximum caption length, ending batch")
                break


        for reference in caps:
            encoded_reference = [w for w in reference.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
            references[0].append(decodeCaption(encoded_reference, idx2word))

        for hyp in predictions:
            pass
           # print(hyp.squeeze(1).tolist())
            #hyp = hyp.squeeze(1)
            #print(hyp.shape)
            #print(hypothesis.view(1,350))
            #encoded_hypothesis = [w for w in hyp.tolist() }]
            hypotheses.append(decodeCaption(hyp.squeeze(1).tolist(), idx2word))

        #print("REFS: ", references)
        #print("HIPS: ", hypotheses)
        assert len(references[0]) == len(hypotheses)
        #break


    return references, hypotheses


if __name__ == '__main__':
    main()
