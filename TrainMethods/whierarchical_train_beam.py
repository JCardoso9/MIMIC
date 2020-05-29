import sys
sys.path.append('../Utils/')
sys.path.append('../')

from setupEnvironment import *
from argParser import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

from nlgeval import NLGEval
import numpy

BEAM_SIZE = 4
MAX_CAPTION_LENGTH = 350
MAX_SENTENCE_LENGTH = 64
MAX_REPORT_LENGTH = 9
DISABLE_TEACHER_FORCING = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


with open("/home/jcardoso/MIMIC/valReportLengthInSentences.json") as json_file:
    trainReportLengthInSentences = json.load(json_file)

with open("/home/jcardoso/MIMIC/valSentencesLengths.json") as json_file:
    trainSentencesLengths = json.load(json_file)

with open("/home/jcardoso/MIMIC/trainSentences.json") as json_file:
    trainSentences = json.load(json_file)



def main():
    argParser = get_args()

    print(argParser)

    modelInfo = None
    classifierInfo = None

    if (argParser.checkpoint is not None):
        modelInfo = torch.load(argParser.checkpoint)

    if (argParser.use_classifier_encoder) and modelInfo is None:
        classifierInfo = torch.load(argParser.classifier_checkpoint)


    encoder, decoder = setupEncoderDecoder(argParser, modelInfo, classifierInfo)

    encoder_optimizer, decoder_optimizer = setupOptimizers(encoder, decoder, argParser, modelInfo)

    decoder_scheduler, encoder_scheduler = setupSchedulers(encoder_optimizer, decoder_optimizer, argParser)

    criterion = setupCriterion(argParser.loss)

    trainLoader, valLoader = setupDataLoaders(argParser)

    # Load word <-> embeddings matrix index correspondence dictionaries
    idx2word, word2idx = loadWordIndexDicts(argParser)

    # Create NlG metrics evaluator
    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])


    print("done")

    
    references, hypotheses = train(argParser, encoder, decoder, trainLoader, word2idx, idx2word)


    #metrics_dict = nlgeval.compute_metrics(references, hypotheses)

    #refs_path, preds_path = save_references_and_predictions(references, hypotheses, argParser.model_name, "Beam")

    #with open('../Experiments/' + argParser.model_name + "/BeamTestResults.txt", "w+") as file:
      #for metric in metrics_dict:
        #file.write(metric + ":" + str(metrics_dict[metric]) + "\n")

def train(argParser, encoder, decoder, trainLoader, word2idx, idx2word):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    decoder.set_teacher_forcing_usage(DISABLE_TEACHER_FORCING)
    decoder.eval()
    encoder.eval()

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # references = [[ref1a, ref1b, ref1c]], hypotheses = [hyp1, hyp2, ...]
    references = [[]]
    hypotheses = list()
    vocab_size = decoder.embedding.weight.shape[0]


    # For each image
    for i, (image, caps, caplens, studies) in enumerate(trainLoader):

        print(studies)
        print(caps.shape)
        print(caps)
        print("-----------------------------")
        batch_size = 2
        report_sentences = []
        report_lengths = []
        sentences_lengths = []
        for study in studies:
            report_sentences.append(trainSentences[study])
            sentences_lengths.append(trainSentencesLengths[study])
            report_lengths.append(trainReportLengthInSentences[study])
           

        for i in range(len(sentences_lengths)):
            sentences_lengths[i] = torch.LongTensor(sentences_lengths[i]) + 2

        sentences_lengths = pad_sequence(sentences_lengths)

        report_lengths = torch.LongTensor(report_lengths)

        print(report_sentences)
        print(report_lengths)
        print(sentences_lengths)       


         #-------------------MODEL OPERATING----------------------------------------
         # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out, pred_logits = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.shape[0]

        label_probs = torch.sigmoid(pred_logits)


        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        resized_img = decoder.resize_encoder_features(encoder_out.mean(dim=1))
        #------------------------------------------------------------------------------
        #print(resized_img.shape)

        losses = torch.zeros(batch_size)

        for t_s in range(max(report_lengths)):

            unfinished_reports = sum([l > t_s for l in report_lengths])
            #print("Sentence :", t_s)
            #print("Unfinished report :", unfinished_reports)

            sorted_report_lengths, sort_ind = report_lengths.sort(dim=0, descending=True)

            #print("Sentences LEnght:", sentences_lengths[t_s])

            sentences_list = []
            for i in range(unfinished_reports):
                sentences_list.append(report_sentences[sort_ind[i]][t_s])

            sentences_list = [torch.LongTensor(i) for i in sentences_list]
            sentences_list = pad_sequence(sentences_list, batch_first=True, padding_value=word2idx['<pad>'])
            print(sentences_list)


            sentences_lengths_copy = sentences_lengths[t_s].clone()
            sentences_lengths_copy = sentences_lengths_copy[sort_ind]

            #print("SORTED SENT LENGTHS:", sentences_lengths_copy)
            sorted_sentence_lengths, sort_ind_sent = sentences_lengths_copy.sort(dim=0, descending=True)

            #print("SORTED SENTENCES:", sentences_list[sort_ind_sent[:unfinished_reports]])
            sentences_list = sentences_list[sort_ind_sent[:unfinished_reports]].to(device)

            #----------------------------MODEL OPERATING-------------------------------------------------------------------
            #encoder_out_copy = encoder_out.clone()
            #label_probas_copy = label_probas.clone()
            #encoder_out_copy = encoder_out_copy[sort_ind]
            #label_probas_copy = label_probas_copy[sort_ind]

            h_sent, c_sent = decoder.init_sent_hidden_state(unfinished_reports)   # (unfinished_reports, hidden_dim

            attention_weighted_visual_encoding, alpha = decoder.visual_attention(encoder_out[sort_ind[:unfinished_reports]], h_sent[:unfinished_reports])  

            print("AWEV",attention_weighted_visual_encoding.shape)

            attention_weighted_label_encoding, alpha = decoder.label_attention(label_probs[sort_ind[:unfinished_reports]], h_sent[:unfinished_reports])

            print("AWEL",attention_weighted_label_encoding.shape)
            print(torch.cat((attention_weighted_visual_encoding, attention_weighted_label_encoding), 1).shape)

            context_vector = decoder.context_vector(torch.cat([attention_weighted_visual_encoding, attention_weighted_label_encoding]
                              ,dim= 1))

            print("context_vector :", context_vector.shape)

            h_sent, c_sent = decoder.sentence_decoder(context_vector, (h_sent[:unfinished_reports], c_sent[:unfinished_reports]))

            topic_vector, stop_prob = decoder.sentence_decoder_fc(decoder.dropout(h_sent)).split(decoder.sentence_decoder_fc_sizes, 1)

            print("topic_vector :", topic_vector.shape)
            print("stop prob:", stop_prob.shape)
            #-------------------------------------------------------------------------------------------

            input = topic_vector
            print("FIrst inputs: ", input)

            predictions = torch.zeros(batch_size, max(sentences_lengths[t_s]), vocab_size).to(device)
            #print("PREDS SHAPE", predictions.shape)

            h_word, c_word = decoder.init_word_hidden_state(unfinished_reports)
            h_word, c_word = decoder.word_decoder(input, (h_word, c_word))


            for  t_w in range(max(sentences_lengths[t_s])):

                 unfinished_sentences = sum([l > t_w for l in sorted_sentence_lengths])

                 #print("SORT IND", sort_ind[:unfinished_sentences])
                 #print("SORT_IND_SENT", sort_ind_sent[:unfinished_sentences])
                 input = decoder.embedding(sentences_list[:unfinished_sentences, t_w])

                 h_word, c_word = decoder.word_decoder(input, (h_word[:unfinished_sentences], c_word[:unfinished_sentences]))

                 print("INPUTS:",input.shape)


            #for sentence_idx in range(len(sentences_list)):
                #print("Target shape:", sentences_list[sentence_idx, :sorted_sentence_lengths[sentence_idx]].shape)
                #print("Preds shape:", predictions[sentence_idx, :sorted_sentence_lengths[sentence_idx], :].shape)



            #losses[sort_ind[sort_ind_sent[:unfinished_sentences]]] +=1 
            #losses[sort_ind[:unfinished_reports]] += 1
            #print("LOSSES", losses)


            break


        break


        while 1 >2:
            sys.exit()

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            if (argParser.model == "Softmax"):
                scores = decoder.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

            elif (argParser.model == "Continuous"):
                preds = decoder.fc(h) #(s, embed_dim)
                preds = nn.functional.normalize(preds, p=2, dim=1)

                # With normalized vectors dot product = cosine similarity
                scores = torch.mm(preds, decoder.embedding.weight.T)  #(s,vocab_size)


            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <eoc>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word2idx['<eoc>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > MAX_CAPTION_LENGTH:
                break
            step += 1

        # If MAX CAPTION LENGTH has been reached and no sequence has reached the eoc
        # there will be no complete seqs, use the incomplete ones
        if k == beam_size:
            complete_seqs.extend(seqs[[incomplete_inds]].tolist())
            complete_seqs_scores.extend(top_k_scores[[incomplete_inds]])

        # Choose best sequence overall
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        ref = [w for w in caps[0].tolist() if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
        references[0].append(decodeCaption(ref, idx2word))

        # Hypotheses
        seq = [w for w in seq if w not in {word2idx['<sos>'], word2idx['<eoc>'], word2idx['<pad>']}]
        hypotheses.append(decodeCaption(seq, idx2word))

        #print("REFS: ", references)
        #print("HIPS: ", hypotheses)
        assert len(references[0]) == len(hypotheses)
        #break

    return references, hypotheses


if __name__ == '__main__':
    main()
