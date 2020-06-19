

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

MAX_SENTENCE_LENGTH = 64
MAX_REPORT_LENGTH = 9
DISABLE_TEACHER_FORCING = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


def main():
    argParser = get_args()

    print(argParser)

    if (argParser.checkpoint is not None):
        modelInfo = torch.load(argParser.checkpoint)

    # Load model
    encoder, decoder = setupEncoderDecoder(argParser, modelInfo)

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

    print(metrics_dict)

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
    for i, (image, caps, caplens,_) in enumerate(
            tqdm(testLoader, desc="Evaluating hierarchical model with greedy decoding: ")):

        caps = caps.to(device)

        # Move to GPU device, if available
        image = image.to(device)  # (batch_size, 3, 224, 224)

        # Encode
        encoder_out, pred_logits = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.shape[0]

        label_probs = torch.sigmoid(pred_logits)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        resized_img = decoder.resize_encoder_features(encoder_out.mean(dim=1))

        t_s = 0

        sentence = []
        h_sent, c_sent = decoder.init_sent_hidden_state(batch_size)

        words = 0
        while t_s < MAX_REPORT_LENGTH:

             attention_weighted_visual_encoding, visual_alpha = decoder.visual_attention(encoder_out, h_sent)

             #attention_weighted_label_encoding, label_alpha = decoder.label_attention(label_probs, h_sent)

             #context_vector = decoder.context_vector_fc(torch.cat([attention_weighted_visual_encoding, attention_weighted_label_encoding]
             #                 ,dim= 1))

             context_vector = decoder.context_vector_fc(attention_weighted_visual_encoding)

             #print(context_vector)
             #print(h_sent)

             prev_h_sent = h_sent

             h_sent, c_sent = decoder.sentence_decoder(context_vector, (h_sent, c_sent))

             topic_vector = decoder.topic_vector(decoder.tanh(decoder.context_vector_W_h_t(h_sent) + decoder.context_vector_W(context_vector)))

             #print(topic_vector)


             stop_prob = decoder.stop(decoder.tanh(decoder.stop_h_1(prev_h_sent) + decoder.stop_h(h_sent)))

             #topic_vector, stop_prob = decoder.sentence_decoder_fc(h_sent).split(decoder.sentence_decoder_fc_sizes, 1)

             #print(topic_vector)

             #print("stop prob:", torch.sigmoid(stop_prob))

             if (torch.sigmoid(stop_prob) > 0.5):
                 break

             #predictions = torch.zeros(unfinished_reports, MAX, vocab_size).to(device)
             h_word, c_word = decoder.init_word_hidden_state(batch_size)
             #h_word, c_word = decoder.word_decoder(topic_vector, (h_word, c_word))
             #word_lstm_input = topic_vector
             word_lstm_input = decoder.sos_embedding.expand(batch_size, decoder.embed_dim).to(device)
             t_w = 0
             while t_w < MAX_SENTENCE_LENGTH: 

                 #print(h_word)
                 #print(word_lstm_input)
                 h_word, c_word = decoder.word_decoder(torch.cat([topic_vector, word_lstm_input],1), (h_word, c_word))

                 if (argParser.model == "HierarchicalSoftmax"):
                     
                     scores = decoder.fc(h_word)  # (batch_size, vocab_size)
                     #print("scores:", scores)
                     scores = F.log_softmax(scores, dim=1)   # (batch_size, vocab_size)
                     #print("softmaxed", scores)
                     pred_word_indexes = torch.argmax(scores, dim=1)


                 elif (argParser.model == "HierarchicalContinuous"):
                     preds = decoder.fc(h_word) # (batch_size, embed_dim)
                     preds =  torch.nn.functional.normalize(preds, p=2, dim=1)

                     cosine_sim_scores = torch.mm(preds, decoder.embedding.weight.T)  #(batch_size, vocab_size)

                     pred_word_indexes = torch.argmax(cosine_sim_scores, dim=1)  #(ba



                 sentence.append(pred_word_indexes.item())
                 #print(sentence)

                 if (t_w > 1 and pred_word_indexes.item() == word2idx['.']):
                     break
 
                 t_w += 1
                 #if (t_w > 2):
                 word_lstm_input = decoder.embedding(pred_word_indexes)
                 #else:
                 #word_lstm_input = decoder.embedding(caps[:,words]).to(device)
                 #words += 1
                 #print(caps[:,words])

             t_s +=1


        #print(caps)

        #print(sentence)
        hypotheses.append(decodeCaption(sentence, idx2word))
        #print("HIPS" ,hypotheses)

        for reference in caps:
            #print("Caps", caps)
            encoded_reference = [w for w in reference.tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
            references[0].append(decodeCaption(encoded_reference, idx2word))
        #print("REFS:", references[0])

        assert len(references[0]) == len(hypotheses)


        #break



    return references, hypotheses


if __name__ == '__main__':
    main()
