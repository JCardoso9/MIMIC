

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
DISABLE_TEACHER_FORCING = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


# NOTE: This code can be easily changed to support batched operations
# however the program allocates insurmountable amounts of memory during the for/while
# cycle during the decoding process in the decode_step line. The program is unable to handle smaller batches
# when compared to the training phase, even though during testing the gradients are not kept...

# After some time looking into this we decided to just
# tank the extended validation time for unbatched operations but if a solution arises
# we would look forward to hear it. In any case the needed code to change to batched operations is left commented


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

    # Generate predictions using greedy decoding
    references, hypotheses = evaluate_greedy(argParser, encoder, decoder, testLoader, word2idx, idx2word)

    # Evaluate generated sentences vs references
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
    decoder.set_teacher_forcing_usage(DISABLE_TEACHER_FORCING)
    decoder.eval()
    encoder.eval()

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # references = [[ref1a, ref1b, ref1c]], hypotheses = [hyp1, hyp2, ...]
    references = [[]]
    hypotheses = list()
    vocab_size = decoder.embedding.weight.shape[0]

    # For each image
    for i, (image, caps, caplens) in enumerate(
            tqdm(testLoader, desc="Evaluating with greedy decoding: ")):

        t = 0

        #finished_sequences = []      NEEDED FOR BATCHED OPERATIONS (BO)
        nr_ended_sequences = 0     #REMOVE IF USING BO

        # Move to GPU device, if available
        image = image.to(device)  # (batch_size, 3, 224, 224)

        # Encode
        if (argParser.use_classifier_encoder):
            encoder_out, _ = encoder(image)
        else:
            encoder_out = encoder(image)
        #encoder_out = encoder(image)  # (batch_size, enc_image_size, enc_image_size, encoder_dim)

        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        h, c = decoder.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)


        input = decoder.sos_embedding.expand(batch_size, decoder.embed_dim).to(device) # (batch_size, embed_dim)

        predictions = torch.zeros(batch_size, MAX_CAPTION_LENGTH, 1, dtype=torch.long).to(device)
        alphas = torch.zeros(batch_size, MAX_CAPTION_LENGTH , num_pixels).to(device)



        #while batch_size > len(finished_sequences)   NEEDED FOR BO
        while batch_size > nr_ended_sequences:   # REMOVE IF USING BO

            awe, alpha = decoder.attention(encoder_out, h)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (batch_size, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([input, awe], dim=1), (h, c))  # (batch_size, decoder_dim)

            if (argParser.model == "Softmax"):
                scores = decoder.fc(h)  # (batch_size, vocab_size)
                scores = F.log_softmax(scores, dim=1)   # (batch_size, vocab_size)
                pred_word_indexes = torch.argmax(scores, dim=1)   # (batch_size, 1)

                # Since we keep decoding even after producing an eoc, this only works if using a single instance
                # During batched operations sequences that have already produced eoc can (and most likely will)
                # produce <eoc>s again
                nr_ended_sequences += pred_word_indexes.tolist().count(word2idx['<eoc>'])  #REMOVE IF USING BO


                #NEEDED FOR BO
                #sentences_that_produced_eoc = [i for i, e in enumerate(pred_word_indexes.tolist()) if e == word2idx['<eoc>']]
                #finished_sequences.extend([i for i in sentences_that_produced_eoc if i not in finished_sequences])


            elif (argParser.model == "Continuous"):
                preds = decoder.fc(h) # (batch_size, embed_dim)
                preds =  torch.nn.functional.normalize(preds, p=2, dim=1)

                # With normalized vectors  dot product = cosine similarity 
                cosine_sim_scores = torch.mm(preds, decoder.embedding.weight.T)  #(batch_size, vocab_size)
                pred_word_indexes = torch.argmax(cosine_sim_scores, dim=1)  #(batch_size, 

                # See softmax model comments
                nr_ended_sequences += pred_word_indexes.tolist().count(word2idx['<eoc>'])  # REMOVE IF USING BO

                # NEEDED FOR BO
                # sentences_that_produced_eoc = [i for i, e in enumerate(pred_word_indexes.tolist()) if e == word2idx['<eoc>']]
                # finished_sequences.extend([i for i in sentences_that_produced_eoc if i not in finished_sequences])


            predictions[:, t, :] = pred_word_indexes
            alphas[:, t, :] = alpha

            input = decoder.embedding(pred_word_indexes)
            t +=1
      
            if t > MAX_CAPTION_LENGTH - 1:
                #print("Reached maximum caption length, ending batch")
                break


        # The decodeCaption function stops after the first <eoc>, but does not handle pads or sos
        for reference in caps:
            encoded_reference = [w for w in reference.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
            references[0].append(decodeCaption(encoded_reference, idx2word))

        for hyp in predictions:
            hypotheses.append(decodeCaption(hyp.squeeze(1).tolist(), idx2word))

        #print("REFS: ", references)
        #print("HIPS: ", hypotheses)
        assert len(references[0]) == len(hypotheses)
        #break


    return references, hypotheses


if __name__ == '__main__':
    main()
