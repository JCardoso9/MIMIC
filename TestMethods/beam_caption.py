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

BEAM_SIZE = 4
MAX_CAPTION_LENGTH = 350
DISABLE_TEACHER_FORCING = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


obj_study= '55609649'

def main():
    argParser = get_args()

    print(argParser)

    print(argParser.checkpoint)
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

    vocab_size = decoder.vocab_size

    references, hypotheses = evaluate_beam(argParser, BEAM_SIZE, encoder, decoder, testLoader, word2idx, idx2word)


    metrics_dict = nlgeval.compute_metrics(references, hypotheses)

    refs_path, preds_path = save_references_and_predictions(references, hypotheses, argParser.model_name, "Beam")

    with open('../Experiments/' + argParser.model_name + "/BeamTestResults.txt", "w+") as file:
      for metric in metrics_dict:
        file.write(metric + ":" + str(metrics_dict[metric]) + "\n")

def evaluate_beam(argParser, beam_size, encoder, decoder, testLoader, word2idx, idx2word):
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
    for i, (image, caps, caplens, studies) in enumerate(
            tqdm(testLoader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        #print(studies)

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        if (argParser.use_classifier_encoder):
            encoder_out, _ = encoder(image)
        else:
            encoder_out = encoder(image)

        #encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word2idx['<sos>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        #print("Seqs1:",seqs.shape)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

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

#        print("Current",studies[0])
#        print("obj",obj_study)
#        if (studies[0] == obj_study):
#            print("Refs:", decodeCaption(ref, idx2word))
#            print("HIPS: ", decodeCaption(seq, idx2word))
#            break
#        print("REFS: ", references)
#        print("HIPS: ", hypotheses)
        assert len(references[0]) == len(hypotheses)
#        break

    return references, hypotheses


if __name__ == '__main__':
    main()
