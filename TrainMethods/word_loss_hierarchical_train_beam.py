import sys
sys.path.append('../Utils/')
sys.path.append('../')
sys.path.append('../TestMethods/')

from setupEnvironment import *
from argParser import *
from hierarchical_greedy_caption import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from nlgeval import NLGEval
import numpy
import time


BEAM_SIZE = 4
MAX_CAPTION_LENGTH = 350
MAX_SENTENCE_LENGTH = 64
MAX_REPORT_LENGTH = 9
DISABLE_TEACHER_FORCING = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


with open("/home/jcardoso/MIMIC/trainReportLengthInSentences.json") as json_file:
    trainReportLengthInSentences = json.load(json_file)

with open("/home/jcardoso/MIMIC/trainSentencesLengths.json") as json_file:
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

    if not os.path.isdir('../Experiments/' + argParser.model_name):
        os.mkdir('../Experiments/' + argParser.model_name)

    trainingEnvironment = TrainingEnvironment(argParser)

    cudnn.benchmark = True


    encoder, decoder = setupEncoderDecoder(argParser, modelInfo, classifierInfo)

    encoder_optimizer, decoder_optimizer = setupOptimizers(encoder, decoder, argParser, modelInfo)

    decoder_scheduler, encoder_scheduler = setupSchedulers(encoder_optimizer, decoder_optimizer, argParser)

    criterion = setupCriterion(argParser.loss)

    binary_criterion = nn.BCEWithLogitsLoss()

    trainLoader, valLoader = setupDataLoaders(argParser)

    # Load word <-> embeddings matrix index correspondence dictionaries
    idx2word, word2idx = loadWordIndexDicts(argParser)

    # Create NlG metrics evaluator
    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

    scheduled_sampling_prob = decoder.scheduled_sampling_prob

    
    for epoch in range(trainingEnvironment.start_epoch, trainingEnvironment.epochs):

        if epoch > 1 and argParser.use_scheduled_sampling and epoch % argParser.scheduled_sampling_decay_epochs == 0:
            scheduled_sampling_prob += argParser.rate_change_scheduled_sampling_prob
            decoder.set_scheduled_sampling_prob(scheduled_sampling_prob)


        if trainingEnvironment.epochs_since_improvement == argParser.early_stop_epoch_threshold:
            break


    
        train(argParser, encoder, decoder, trainLoader, word2idx, idx2word, criterion, encoder_optimizer, decoder_optimizer, binary_criterion, epoch)

 #      references, hypotheses = hierarchical_evaluate_beam(argParser, BEAM_SIZE, encoder, decoder, valLoader, word2idx, idx2word)
        references, hypotheses = evaluate_greedy(argParser, encoder, decoder, valLoader, word2idx, idx2word)

        encoder_scheduler.step()
        decoder_scheduler.step()

        metrics_dict = nlgeval.compute_metrics(references, hypotheses)
        print(metrics_dict)

        with open('../Experiments/' + argParser.model_name + "/metrics.txt", "a+") as file:
            file.write("Epoch " + str(epoch) + " results:\n")
            for metric in metrics_dict:
                file.write(metric + ":" + str(metrics_dict[metric]) + "\n")
            file.write("------------------------------------------\n")

        recent_bleu4 = metrics_dict['CIDEr']

    #     Check if there was an improvement
        is_best = recent_bleu4 > trainingEnvironment.best_bleu4

        trainingEnvironment.best_bleu4 = max(recent_bleu4, trainingEnvironment.best_bleu4)

        print("Best BLEU: ", trainingEnvironment.best_bleu4)
        if not is_best:
            trainingEnvironment.epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (trainingEnvironment.epochs_since_improvement,))
        else:
            trainingEnvironment.epochs_since_improvement = 0


#        recent_bleu4 = 0
#        is_best = True
#        metrics_dict = {}

        # Save checkpoint
        save_checkpoint(argParser.model_name, epoch, trainingEnvironment.epochs_since_improvement, encoder.state_dict(), decoder.state_dict(), encoder_optimizer.state_dict(),
                        decoder_optimizer.state_dict(), recent_bleu4, is_best, metrics_dict, trainingEnvironment.best_loss)


def train(argParser, encoder, decoder, trainLoader, word2idx, idx2word, criterion, encoder_optimizer, decoder_optimizer, binary_criterion, epoch):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """


    decoder.set_teacher_forcing_usage(argParser.use_tf_as_input)
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)

    start = time.time()


    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # references = [[ref1a, ref1b, ref1c]], hypotheses = [hyp1, hyp2, ...]
    references = [[]]
    hypotheses = list()
    vocab_size = decoder.embedding.weight.shape[0]


    # For each image
    for i, (image, caps, caplens, studies) in enumerate(trainLoader):

        data_time.update(time.time() - start)

        #print(studies)
        #print(caps.shape)
        #print(caps)
        #print("-----------------------------")
        report_sentences = []
        report_lengths = []
        sentences_lengths = []
        for study in studies:
            report_sentences.append(trainSentences[study])
            sentences_lengths.append(trainSentencesLengths[study])
            report_lengths.append(trainReportLengthInSentences[study])
           

        for idx in range(len(sentences_lengths)):
            sentences_lengths[idx] = torch.LongTensor(sentences_lengths[idx]) + 1

        sentences_lengths = pad_sequence(sentences_lengths)

        report_lengths = torch.LongTensor(report_lengths)

        #print(report_sentences)
        #print(report_lengths)
        #print(sentences_lengths)       


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

        loss = 0.

        h_sent, c_sent = decoder.init_sent_hidden_state(batch_size)   # (unfinished_reports, hidden_d


        visual_alphas = torch.zeros(batch_size, max(report_lengths), num_pixels).to(device)
        label_alphas = torch.zeros(batch_size, max(report_lengths), label_probs.shape[1]).to(device)

        #print("NEW BATCH")

        #print(report_lengths)
        #print("SUMMED", sum(report_lengths))

        for t_s in range(max(report_lengths)):

            unfinished_reports = sum([l > t_s for l in report_lengths])
            #print("Sentence :", t_s)
            #print("Unfinished report :", unfinished_reports)

            sorted_report_lengths, sort_ind = report_lengths.sort(dim=0, descending=True)

            #print("Sentences LEnght:", sentences_lengths[t_s])
            #print(sort_ind)

            sentences_list = []
            for idx in range(unfinished_reports):
                sentences_list.append(report_sentences[sort_ind[idx]][t_s])

            sentences_list = [torch.LongTensor(i) for i in sentences_list]
            sentences_list = pad_sequence(sentences_list, batch_first=True, padding_value=word2idx['<pad>'])
            #print(sentences_list)


            sentences_lengths_copy = sentences_lengths[t_s].clone()
            sentences_lengths_copy = sentences_lengths_copy[sort_ind]

            #print("SORTED SENT LENGTHS:", sentences_lengths_copy)
            sorted_sentence_lengths, sort_ind_sent = sentences_lengths_copy.sort(dim=0, descending=True)
            #print("SORTED SENTENCES:", sentences_list[sort_ind_sent[:unfinished_reports]])
            sorted_sentence_list = sentences_list[sort_ind_sent[:unfinished_reports]].to(device)
            #----------------------------MODEL OPERATING-------------------------------------------------------------------
            #encoder_out_copy = encoder_out.clone()
            #label_probas_copy = label_probas.clone()
            #encoder_out_copy = encoder_out_copy[sort_ind]
            #label_probas_copy = label_probas_copy[sort_ind]

            #h_sent, c_sent = decoder.init_sent_hidden_state(unfinished_reports)   # (unfinished_reports, hidden_dim

            attention_weighted_visual_encoding, visual_alpha = decoder.visual_attention(encoder_out[sort_ind[:unfinished_reports]], h_sent[:unfinished_reports])

            #print("AWEV",attention_weighted_visual_encoding.shape)
            #print(num_pixels)
            #print(visual_alpha.shape)
            
            #attention_weighted_label_encoding, label_alpha = decoder.label_attention(label_probs[sort_ind[:unfinished_reports]], h_sent[:unfinished_reports])

            #print("LABEL ALHPA SHAPE",label_alpha.shape)

            #print("AWEL",attention_weighted_label_encoding.shape)
            #print(torch.cat((attention_weighted_visual_encoding, attention_weighted_label_encoding), 1).shape)

            #context_vector = decoder.context_vector_fc(torch.cat([attention_weighted_visual_encoding, attention_weighted_label_encoding]
            #                  ,dim= 1))

            context_vector = decoder.context_vector_fc(attention_weighted_visual_encoding)


            #print("context_vector :", context_vector.shape)

            prev_h_sent = h_sent[:unfinished_reports]

            h_sent, c_sent = decoder.sentence_decoder(context_vector, (h_sent[:unfinished_reports], c_sent[:unfinished_reports]))

            topic_vector = decoder.topic_vector(decoder.tanh(decoder.context_vector_W_h_t(h_sent[:unfinished_reports]) + decoder.context_vector_W(context_vector)))

            stop_prob = decoder.stop(decoder.tanh(decoder.stop_h_1(prev_h_sent) + decoder.stop_h(h_sent[:unfinished_reports])))
            #print("OSRT IND SNENT", sort_ind_sent)

            #topic_vector, stop_prob = decoder.sentence_decoder_fc(decoder.dropout(h_sent)).split(decoder.sentence_decoder_fc_sizes, 1)

            #print("topic_vector :", topic_vector.shape)
            #print("stop prob:", stop_prob.shape)
            #-------------------------------------------------------------------------------------------

            #input = topic_vector
            #print("FIrst inputs: ", input)

            predictions = torch.zeros(unfinished_reports, max(sentences_lengths[t_s]), vocab_size).to(device)
            #print("PREDS SHAPE", predictions.shape)

            h_word, c_word = decoder.init_word_hidden_state(unfinished_reports)
            #h_word, c_word = decoder.word_decoder(topic_vector, (h_word, c_word))

            input = decoder.sos_embedding.expand(unfinished_reports, decoder.embed_dim).to(device)

            sorted_sentence_lengths = (sorted_sentence_lengths -1).tolist()
            #print("SENTENCES:",sorted_sentence_list)
            unfinished_sentences = unfinished_reports
            topic_vector = topic_vector[sort_ind_sent[:unfinished_reports]]


            for  t_w in range(max(sorted_sentence_lengths)):

                 #unfinished_sentences = sum([l > t_w for l in sorted_sentence_lengths])

                 #print("SORT IND", sort_ind[:unfinished_sentences])
                 #print("SORT_IND_SENT", sort_ind_sent[:unfinished_sentences])
                 #print(input.shape)
                 #print("topic", topic_vector[:unfinished_sentences].shape)
                 
                 h_word, c_word = decoder.word_decoder(torch.cat([topic_vector[:unfinished_sentences], input],1), (h_word[:unfinished_sentences], c_word[:unfinished_sentences]))

                 preds = decoder.fc(decoder.dropout(h_word))

                 predictions[:unfinished_sentences, t_w, :] = preds

                 unfinished_sentences = sum([l > t_w for l in sorted_sentence_lengths])

                 if t_w > 0 and (random.random() < decoder.scheduled_sampling_prob):
                 #if (t_w > 0):
                    preds = F.log_softmax(preds, dim=1)
                    preds = torch.argmax(preds[:unfinished_sentences], dim=1)
                    #print("Preds:", preds)
                    input = decoder.embedding(preds)
                 else:
                    #print("INPUTS:",sorted_sentence_list[:unfinished_sentences, t_w])
                    input = decoder.embedding(sorted_sentence_list[:unfinished_sentences, t_w])
                 #h_word, c_word = decoder.word_decoder(input, (h_word[:unfinished_sentences], c_word[:unfinished_sentences]))



                 #print("INPUTS:",input.shape)

                 #preds = decoder.fc(decoder.dropout(h_word))

                 #predictions[:unfinished_sentences, t_w, :] = preds

                 #preds1 = F.log_softmax(preds, dim=1)
                 #preds1 = torch.argmax(preds1[:unfinished_sentences], dim=1)
                 #print("Preds:", preds1)
            #sys.exit()
            
            #for sentence_idx in range(len(sorted_sentence_list)):
                #sorted_sentence_lengths =  sorted_sentence_lengths.tolist()
            #print(predictions.shape)
            #print(sorted_sentence_list)
            #print(sorted_sentence_list.shape)

            stop_targets = torch.zeros(unfinished_reports, dtype=torch.float).to(device)
            stop_targets[ (t_s +1) == sorted_report_lengths[:unfinished_reports]] = 1.
            #print("Stop Targets" , stop_targets)
            stop_targets.unsqueeze_(1)
            #print("Stop Targets shape", stop_targets.shape )

            #print(t_s)
            #print(sorted_report_lengths)
            #stop_prob_target = ((t_s + 1) / 1.0) / sorted_report_lengths[:unfinished_reports].type(torch.FloatTensor)
            #print("STOP PROB TARGETS: ", stop_prob_target)

            #print("PREDICTED STOP PROBS:", stop_prob)
            #print(stop_prob.shape)
            #stop_prob = decoder.sigmoid(stop_prob)
            #print("SOFTMAXED STOP PROBS:", stop_prob)
            #print("SOFTMAXED STOP PROBS shape:", stop_prob.shape)

            #print("stop prob loss:" , binary_criterion(stop_prob, stop_targets))
            loss += 5* binary_criterion(stop_prob, stop_targets)
            #print(predictions.shape)
#            if i % 500 == 0:
#                #print(F.log_softmax(predictions[:unfinished_reports,:,:], dim=1))
#                print("PREDS SHAPE:", torch.argmax(F.log_softmax(predictions[:unfinished_reports,:,:], dim=1), dim=2).shape)
#                print("PREDS",torch.argmax(F.log_softmax(predictions[:unfinished_reports,:,:], dim=1), dim=2))
#                print("---------------------------------------------------------")
#                print("TARGETS SHAPE",sorted_sentence_list.shape)
#                print("targets ", sorted_sentence_list)

            total_loss = 0.

            for sentence_idx in range(unfinished_reports):
                #print("preds.shape", predictions.shape)
                #print("preds", predictions[sentence_idx, :sorted_sentence_lengths[sentence_idx]])
                #print("preds shape", predictions[sentence_idx, :sorted_sentence_lengths[sentence_idx]].shape)

                #print("targets", sorted_sentence_list[sentence_idx, :sorted_sentence_lengths[sentence_idx]])
                #print("targets shape", sorted_sentence_list[sentence_idx, :sorted_sentence_lengths[sentence_idx]].shape)
                sentences_loss = criterion(predictions[sentence_idx, :sorted_sentence_lengths[sentence_idx]], sorted_sentence_list[sentence_idx, :sorted_sentence_lengths[sentence_idx]])
                total_loss += sentences_loss
#                print("total_loss", total_loss)
#                print("sentences_loss ", sentences_loss)
#                print("total loss after div", total_loss / unfinished_reports)

          
            loss += total_loss / unfinished_reports 
#            print("loss" , loss)

            #predictions = pack_padded_sequence(predictions, sorted_sentence_lengths[:unfinished_reports], batch_first=True)
            #print("WIthout sos",sorted_sentence_list[:, 1:])
            #targets = pack_padded_sequence(sorted_sentence_list, sorted_sentence_lengths[:unfinished_reports], batch_first=True)
            #loss += criterion(predictions.data, targets.data)
            #print("LOSSC", lossc)

            #loss += lossc

            visual_alphas[:unfinished_reports, t_s, :] = visual_alpha
            #label_alphas[:unfinished_reports, t_s, :] = label_alpha

  #          print("LOSSSSS", loss)

            #loss +=  argParser.alpha_c * ((1. - visual_alphas.sum(dim=1)) ** 2).mean()

 #           print("L WITH VISUAL" , loss)
            #loss +=  argParser.alpha_c * ((1. - label_alphas.sum(dim=1)) ** 2).mean() 

#            print("L WITH ALL",loss)
                #print("Target shape:", sentences_list[sentence_idx, :sorted_sentence_lengths[sentence_idx]].shape)
                #print("Preds shape:", predictions[sentence_idx, :sorted_sentence_lengths[sentence_idx], :].shape)



            #losses[sort_ind[sort_ind_sent[:unfinished_sentences]]] +=1 
            #losses[sort_ind[:unfinished_reports]] += 1
            #print("LOSSES", losses)


            #break

        #print("LOSS:", loss)

        #loss = loss / max(report_lengths)

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()


#        if argParser.grad_clip is not None:
#            clip_gradient(decoder_optimizer, argParser.grad_clip)
#            if encoder_optimizer is not None:
#                clip_gradient(encoder_optimizer, argParser.grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()




         # Keep track of metrics
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % argParser.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(trainLoader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))

    path = '../Experiments/' + argParser.model_name + '/trainLosses.txt'
    writeLossToFile(losses.avg, path)
        #break



if __name__ == '__main__':
    main()
