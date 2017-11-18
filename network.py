# ref: https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LRN(nn.Module):

    def __init__(self, k, n, alpha, beta):
        super(LRN, self).__init__()
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.weights = torch.FloatTensor(1, 1, self.n, 1, 1).fill_(1)
        self.weights = nn.Parameter(self.weights)
        self.padding = (int((self.n - 1) / 2), 0, 0)

    def forward(self, x):
        denom = x.pow(2).unsqueeze(1)
        denom = F.conv3d(denom, self.weights, padding=self.padding)
        denom = denom.squeeze(1).mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(denom)
        return(x)

# This implementation of LRN doesn't work because AvgPool3D with padding is not in stable release yet.
# ref: https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
# class LRN(nn.Module):
#     def __init__(self, k=2, n=1, alpha=1.0, beta=0.75):
#         super(LRN, self).__init__()
#         self.average = nn.AvgPool3d(kernel_size=(n, 1, 1), stride=1, padding=(int((n - 1.0) / 2), 0, 0))
#         self.k = k
#         self.alpha = alpha
#         self.beta = beta

#     def forward(self, x):
#         div = x.pow(2).unsqueeze(1)
#         div = self.average(div).squeeze(1)
#         div = div.mul(self.alpha).add(self.k).pow(self.beta)
#         x = x.div(div)
#         return x


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
        # NOTE: LRN has been disabled as it didn't seem to improve performance but took longer to train.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            self.activation(relu),
            # LRN(k=2, n=5, alpha=1e-4, beta=0.75),
            self.max_pool(overlap),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            self.activation(relu),
            # LRN(k=2, n=5, alpha=1e-4, beta=0.75),
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
                    print('{}: {}'.format(i, m))
                    m.weight.data.normal_(0, 0.01)
                    if i in [3, 8, 10]:
                        m.bias.data.fill_(1.0)
            for i, m in enumerate(self.classifier.children()):
                if type(m) == nn.Linear:
                    print('{}: {}'.format(i, m))
                    m.weight.data.normal_(0, 0.01)
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
