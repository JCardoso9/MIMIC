
import sys
sys.path.append('../Utils/')

from setupEnvironment import *
from argParser import *
from TrainingEnvironment import *
from generalUtilities import *
from train import *
from val import  *

import torch
import torch.backends.cudnn as cudnn

from datetime import datetime

from nlgeval import NLGEval



def main():

    print("Starting training process MIMIC")

    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'GreedyMatchingScore', 'VectorExtremaCosineSimilarity', 'EmbeddingAverageCosineSimilarity'])

    argParser = get_args()

    if not os.path.isdir('../Experiments/' + argParser.model_name):
        os.mkdir('../Experiments/' + argParser.model_name)

    trainingEnvironment = TrainingEnvironment(argParser)

    encoder, decoder, criterion, embeddings, encoder_optimizer, decoder_optimizer, dec_scheduler, enc_scheduler, idx2word, word2idx = setupModel(argParser)

    # Create data loaders
    trainLoader, valLoader = setupDataLoaders(argParser)

    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


    for epoch in range(trainingEnvironment.start_epoch, trainingEnvironment.epochs):
        # Decay learning rate if there is no improvement for "decay_LR_epoch_threshold" consecutive epochs,
        #  and terminate training after minimum LR has been achieved and  "early_stop_epoch_threshold" epochs without improvement
        if trainingEnvironment.epochs_since_improvement == argParser.early_stop_epoch_threshold:
            break
#        if trainingEnvironment.epochs_since_improvement >= argParser.decay_LR_epoch_threshold and trainingEnvironment.current_lr > argParser.lr_threshold:
            #trainingEnvironment.current_lr = adjust_learning_rate(decoder_optimizer, 0.5)
            #if argParser.fine_tune_encoder:
             #   adjust_learning_rate(encoder_optimizer, 0.5)

        # One epoch's training
        train(argParser,train_loader=trainLoader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        references, hypotheses, recent_loss = validate(argParser,val_loader=valLoader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                idx2word=idx2word,
                                embeddings=embeddings)

        enc_scheduler.step()
        dec_scheduler.step()

        # nlgeval = NLGEval()
        metrics_dict = nlgeval.compute_metrics(references, hypotheses)

        print("Metrics: " , metrics_dict)
        with open('../Experiments/' + argParser.model_name + "/metrics.txt", "a+") as file:
            file.write("Epoch " + str(epoch) + " results:\n")
            for metric in metrics_dict:
                file.write(metric + ":" + str(metrics_dict[metric]) + "\n")
            file.write("------------------------------------------\n")

        recent_bleu4 = metrics_dict['CIDEr']

        # Check if there was an improvement
        is_best = recent_bleu4 > trainingEnvironment.best_bleu4

        trainingEnvironment.best_bleu4 = max(recent_bleu4, trainingEnvironment.best_bleu4)

        print("Best BLEU: ", trainingEnvironment.best_bleu4)
        if not is_best:
            trainingEnvironment.epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (trainingEnvironment.epochs_since_improvement,))
        else:
            trainingEnvironment.epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(argParser.model_name, epoch, trainingEnvironment.epochs_since_improvement, encoder.state_dict(), decoder.state_dict(), encoder_optimizer.state_dict(),
                        decoder_optimizer.state_dict(), recent_bleu4, is_best, metrics_dict, trainingEnvironment.best_loss)





if __name__ == "__main__":
    main()


