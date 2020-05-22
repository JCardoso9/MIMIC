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

BEAM_SIZE = 4
MAX_CAPTION_LENGTH = 350
MAX_SENTENCE_LENGTH = 64
MAX_REPORT_LENGTH = 9
DISABLE_TEACHER_FORCING = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


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
    for i, (image, caps, caplens, sentences) in enumerate(trainLoader):


        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out, pred_logits = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.shape[0]

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k

        # Tensor to store top k previous words at each step; now they're just <start>

        #print(encoder_out.mean(dim=1).shape)

        resized_img = decoder.resize_encoder_features(encoder_out.mean(dim=1))

        #print(resized_img.shape)

        nr_sentences = 0
        h_sent, c_sent = decoder.init_sent_hidden_state(encoder_out)

        #print(h_sent.shape)
        #print(c_sent.shape)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while nr_sentences < MAX_REPORT_LENGTH:


            print(pred_logits)
            print("------\n")
            print(torch.sigmoid(pred_logits))
            print(pred_logits.shape)
            print("Enc OUT",encoder_out.shape)
            attention_weighted_visual_encoding, alpha = decoder.visual_attention(encoder_out,
                                                                h_sent)  #ADD BATCH!!!!!!

            print("AWEV",attention_weighted_visual_encoding.shape)

            attention_weighted_label_encoding, alpha = decoder.label_attention(pred_logits,
                                                                h_sent)

            print("AWEL",attention_weighted_label_encoding.shape)


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
