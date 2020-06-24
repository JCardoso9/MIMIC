
import sys
sys.path.append('../Dataset/')
sys.path.append('../Utils/')
sys.path.append('../Models/')
sys.path.append('../../')

from setupEnvironment import *
from argParser import *
from TrainingEnvironment import *
from generalUtilities import *

from ClassifyingEncoder import *
from ClassXRayDataset import *
import torch
import torch.backends.cudnn as cudnn

from datetime import datetime
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    print("Starting training process MIMIC")

    argParser = get_args()

    if not os.path.isdir('../Experiments/' + argParser.model_name):
        os.mkdir('../Experiments/' + argParser.model_name)


    trainingEnvironment = TrainingEnvironment(argParser)

    encoder = ClassifyingEncoder()
    
    encoder.to(device)

    encoder.fine_tune(argParser.fine_tune_encoder)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=argParser.encoder_lr) if argParser.fine_tune_encoder else None

    if (argParser.checkpoint is not None):
      decoder_optimizer.load_state_dict(modelInfo['decoder_optimizer'])
      encoder_optimizer.load_state_dict(modelInfo['encoder_optimizer'])

    enc_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, argParser.lr_decay_epochs, argParser.lr_decay)

    criterion = nn.BCEWithLogitsLoss().to(device)


    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


    for epoch in range(trainingEnvironment.start_epoch, trainingEnvironment.epochs):

        # Decay learning rate if there is no improvement for "decay_LR_epoch_threshold" consecutive epochs,
        #  and terminate training after minimum LR has been achieved and  "early_stop_epoch_threshold" epochs without improvement
        if trainingEnvironment.epochs_since_improvement == argParser.early_stop_epoch_threshold:
            break

        # One epoch's training
        _ = runEpochs(epoch, "Train", encoder, criterion, encoder_optimizer, argParser)

        recent_loss = runEpochs(epoch, "Test", encoder, criterion, encoder_optimizer, argParser)

        enc_scheduler.step()

        # Check if there was an improvement
        is_best = recent_loss < trainingEnvironment.best_loss

        trainingEnvironment.best_loss = min(recent_loss, trainingEnvironment.best_loss)

        if not is_best:
            trainingEnvironment.epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (trainingEnvironment.epochs_since_improvement,))
        else:
            trainingEnvironment.epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(argParser.model_name, epoch, trainingEnvironment.epochs_since_improvement, encoder.state_dict(), None, encoder_optimizer.state_dict(),
                        None, None, is_best, None, trainingEnvironment.best_loss)







def runEpochs(epoch, mode, encoder, criterion, encoder_optimizer, argParser):



    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


    if (mode =='Train'):
        encoder.train()
        loader = DataLoader(ClassXRayDataset(argParser.trainImgsPath, argParser.csv_path, transform), batch_size=argParser.batch_size, shuffle=True)



    elif (mode =='Test'):
        loader = DataLoader(ClassXRayDataset( argParser.valImgsPath, argParser.csv_path, transform), batch_size=argParser.batch_size, shuffle=True)




    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()



    for i, (imgs, target_labels) in enumerate(loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
    
        features, pred_labels_logits = encoder(imgs)

        loss = criterion(pred_labels_logits, reshape_target_labels(target_labels).to(device))

        encoder_optimizer.zero_grad()
        loss.backward()

        if argParser.grad_clip is not None:
            clip_gradient(encoder_optimizer, argParser.grad_clip)

        encoder_optimizer.step()

        batch_size = imgs.shape[0]

        losses.update(loss.item(), batch_size)
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % argParser.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))

        del loss
        del imgs
        del target_labels
        del pred_labels_logits
        del features

    path = '../Experiments/' + argParser.model_name + '/' + mode + 'Losses.txt'
    writeLossToFile(losses.avg, path)

    return losses.avg




if __name__ == "__main__":
    main()


