import torch
from torch import nn
import torchvision


class RefactoredEncoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, network_name, encoded_image_size=14):
        super(RefactoredEncoder, self).__init__()
        self.network_name = network_name

        if self.network_name == 'resnet101':
            resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
            modules = list(resnet.children())[:-2]
            self.dim = 2048


        elif self.network_name == 'densenet121':
            densenet = torchvision.models.densenet121(pretrained=True)
            modules = list(list(densenet.children())[0])[:-1]
            self.dim = 1024

        self.net = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.net(images)  # (batch_size, 2048, image_size/32, image_size/32)
        #out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        if self.network_name == 'resnet101':
            for c in list(self.net.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
        elif self.network_name == 'densenet121':
            for c in list(self.net.children())[8:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
