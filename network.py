# ref: https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def activation(self, relu):
        if relu:
            return nn.ReLU(inplace=True)
        else:
            return nn.Tanh()

    def max_pool(self, overlap):
        if overlap:
            return nn.MaxPool2d(kernel_size=3, stride=2)
        else:
            return nn.MaxPool2d(kernel_size=2, stride=2)

    def __init__(self, num_classes=1000, relu=True, dropout=True, overlap=True, init_wts=True):
        super(AlexNet, self).__init__()
        # TODO: add response normalization after 1st and 2nd layer
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            self.activation(relu),
            self.max_pool(overlap),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            self.activation(relu),
            self.max_pool(overlap),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            self.activation(relu),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            self.activation(relu),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            self.activation(relu),
            self.max_pool(overlap),
        )

        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        if init_wts:
            for i, m in enumerate(self.features.children()):
                if type(m) == nn.Conv2d:
                    m.weight.data.normal_(0, 0.001)
                    if i in [3, 8, 10]:
                        m.bias.data.fill_(1.0)
            for i, m in enumerate(self.classifier.children()):
                if type(m) == nn.Linear:
                    m.weight.data.normal_(0, 1.0)
                    m.bias.data.fill_(1.0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
