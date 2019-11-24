#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""criteria.py
Implements different models.
"""

# All credits of Synchronized BN go to Tamaki Kojima(tamakoji@gmail.com)
# (https://github.com/tamakoji/pytorch-syncbn)
# DeeplabV3:  L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam.  Re-
# thinking  atrous  convolution  for  semantic  image  segmenta-
# tion. arXiv preprint arXiv:1706.05587, 2017..

# Source based: https://github.com/speedinghzl/pytorch-segmentation-toolbox
# BN: https://github.com/mapillary/inplace_abn

# PSPNet:  H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia.  Pyramid scene
# parsing network. In CVPR, pages 2881–2890, 2017.
# https://arxiv.org/abs/1612.01105


# Other stuff:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# Deeplab:
# https://github.com/speedinghzl/pytorch-segmentation-toolbox
# https://github.com/speedinghzl/Pytorch-Deeplab
# https://github.com/kazuto1011/deeplab-pytorch
# https://github.com/isht7/pytorch-deeplab-resnet
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/CSAILVision/semantic-segmentation-pytorch


# Pretrained:
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/
# 28aab5849db391138881e3c16f9d6482e8b4ab38/dataset.py
# https://github.com/CSAILVision/sceneparsing
# https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/
# 28aab5849db391138881e3c16f9d6482e8b4ab38
# Input normalization (natural images):
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/
# 28aab5849db391138881e3c16f9d6482e8b4ab38/dataset.py

import threading
import sys
import math
import os
import datetime as dt

from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.nn import functional as F

from tools import check_if_allow_multgpu_mode, announce_msg

sys.path.append("..")

from deeplearning.decision_pooling import WildCatPoolDecision, ClassWisePooling
# lock for threads to protect the instruction that cause randomness and make
# them
thread_lock = threading.Lock()
# thread-safe.

import reproducibility

ACTIVATE_SYNC_BN = True
# Override ACTIVATE_SYNC_BN using variable environment in Bash:
# $ export ACTIVATE_SYNC_BN="True"   ----> Activate
# $ export ACTIVATE_SYNC_BN="False"   ----> Deactivate

if "ACTIVATE_SYNC_BN" in os.environ.keys():
    ACTIVATE_SYNC_BN = (os.environ['ACTIVATE_SYNC_BN'] == "True")

announce_msg("ACTIVATE_SYNC_BN was set to {}".format(ACTIVATE_SYNC_BN))

if check_if_allow_multgpu_mode() and ACTIVATE_SYNC_BN:  # Activate Synch-BN.
    from deeplearning.syncbn import nn as NN_Sync_BN
    BatchNorm2d = NN_Sync_BN.BatchNorm2d
    announce_msg("Synchronized BN has been activated. \n"
                 "MultiGPU mode has been activated. "
                 "{} GPUs".format(torch.cuda.device_count()))
else:
    BatchNorm2d = nn.BatchNorm2d
    if check_if_allow_multgpu_mode():
        announce_msg("Synchronized BN has been deactivated.\n"
                     "MultiGPU mode has been activated. "
                     "{} GPUs".format(torch.cuda.device_count()))
    else:
        announce_msg("Synchronized BN has been deactivated.\n"
                     "MultiGPU mode has been deactivated. "
                     "{} GPUs".format(torch.cuda.device_count()))

ALIGN_CORNERS = True

__all__ = ['resnet18', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet'
                '/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet'
                '/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet'
                 '/resnet101-imagenet.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WildCatClassifierHead(nn.Module):
    """
    A WILDCAT type classifier head.
    `WILDCAT: Weakly Supervised Learning of Deep ConvNets for
    Image Classification, Pointwise Localization and Segmentation`,
    Thibaut Durand, Taylor Mordan, Nicolas Thome, Matthieu Cord.
    """
    def __init__(self, inplans, modalities, num_classes, kmax=0.5, kmin=None,
                 alpha=0.6, dropout=0.0):
        super(WildCatClassifierHead, self).__init__()

        self.num_classes = num_classes

        self.to_modalities = nn.Conv2d(
            inplans, num_classes * modalities, kernel_size=1, bias=True)
        self.to_maps = ClassWisePooling(num_classes, modalities)
        self.wildcat = WildCatPoolDecision(
            kmax=kmax, kmin=kmin, alpha=alpha, dropout=dropout)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        The forward function.
        :param x: input tensor.
        :param seed:
        :param prngs_cuda:
        :return:
        """
        modalities = self.to_modalities(x)
        maps = self.to_maps(modalities)
        scores = self.wildcat(x=maps, seed=seed, prngs_cuda=prngs_cuda)

        return scores, maps


class PoissonHead(WildCatClassifierHead):
    """
    A pooling that is based on the WILDCAT.
    But, it uses the original scores of WILDCAT, and  obtain
    one single score using a dense layer. Then, Compute the per-class scores
    using the Poisson distribution.
    See:
    [1] `C. Beckham and C. Pal. Unimodal probability distributions
         for deep ordinal classification. CoRR, abs/1705.05278, 2017.`
    """
    def __init__(self, inplans, modalities, num_classes, kmax=0.5, kmin=None,
                 alpha=0.6, dropout=0.0, tau=1.):
        """

        :param inplans:
        :param modalities:
        :param num_classes:
        :param kmax:
        :param kmin:
        :param alpha:
        :param dropout:
        :param tau: float > 0.
        """
        announce_msg("Using Poisson hard-wired output.")
        super(PoissonHead, self).__init__(
            inplans=inplans, modalities=modalities, num_classes=num_classes,
            kmax=kmax, kmin=kmin, alpha=alpha, dropout=dropout)

        msg = "`tau` should be float. found {} ...[NOT OK]".format(
            type(tau))
        assert isinstance(tau, float), msg
        msg = "`tau` must be in ]0., inf[. found {} ... [NOT OK]".format(tau)
        assert tau > 0., msg

        self.tau = tau
        self.pool_to_one = nn.Sequential(
            nn.Linear(num_classes, 1, bias=True),
            nn.Softplus()
        )

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        The forward function.
        :param x:
        :param seed:
        :param prngs_cuda:
        :return:
        """
        scores, maps = super(PoissonHead, self).forward(
            x=x, seed=seed, prngs_cuda=prngs_cuda)

        nbr_s = scores.shape[0]
        # compute one positive non-zero score.
        onescore = self.pool_to_one(scores)  # a vector. shape: nbr_samples, 1.

        gridscores = onescore.view(-1, 1).repeat(1, self.num_classes)

        sliceit = torch.arange(0, self.num_classes, dtype=scores.dtype,
                               device=scores.device,  requires_grad=False)
        gridclasses = sliceit.view(1, -1).repeat(nbr_s, 1)

        # compute factorial using Gamma approximation using float.
        # https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/
        # scipy.misc.factorial.html
        # https://discuss.pytorch.org/t/is-there-a-gamma-function-in-pytorch/
        # 17122/2
        #
        # n! = gamma(n+1) = exp(log_gamma(n+1)).
        factorialgridclasses = (gridclasses + 1.).lgamma().exp()

        scores = gridclasses * gridscores.log() - gridscores - \
            factorialgridclasses.log()

        scores = - scores / self.tau  # heat up the scores here.

        return scores, maps


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes,
                 namenet="ResNet", modalities=4, kmax=0.5, kmin=None,
                 alpha=0.6, dropout=0.0, poisson=False, tau=1.):
        """
        Init. function.
        :param block: class of the block.
        :param layers: list of int, number of layers per block.
        :param num_classes: int, number of output classes. must be > 1.
        ============================= WILDCAT ==================================
        :param modalities: int, number of modalities for WILDCAT.
        :param kmax: int or float scalar in ]0., 1.]. The number of maximum
        features to consider.
        :param kmin: int or float scalar. If None, it takes the same value as
        :param kmax. The number of minimal features to consider.
        :param alpha: float scalar. A weight , used to compute the final score.
        :param dropout: float scalar. If not zero, a dropout is performed over
        the min and max selected features.
        :param poisson: Bool. If True, we hard-wire a Poisson distribution at
        the output after Wildcat pooling. See [1].
        :param tau: float > 0.. related to the `poisson` variable.

        [1] `C. Beckham and C. Pal. Unimodal probability distributions
         for deep ordinal classification. CoRR, abs/1705.05278, 2017.`
        """
        assert num_classes > 1, "Number of classes must be > 1 ....[NOT OK]"
        self.num_classes = num_classes
        self.namenet = namenet

        self.inplanes = 128
        super(ResNet, self).__init__()

        # Encoder

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Find out the size of the output.

        if isinstance(self.layer4[-1], Bottleneck):
            in_channel4 = self.layer1[-1].bn3.weight.size()[0]
            in_channel8 = self.layer2[-1].bn3.weight.size()[0]
            in_channel16 = self.layer3[-1].bn3.weight.size()[0]
            in_channel32 = self.layer4[-1].bn3.weight.size()[0]
        elif isinstance(self.layer4[-1], BasicBlock):
            in_channel4 = self.layer1[-1].bn2.weight.size()[0]
            in_channel8 = self.layer2[-1].bn2.weight.size()[0]
            in_channel16 = self.layer3[-1].bn2.weight.size()[0]
            in_channel32 = self.layer4[-1].bn2.weight.size()[0]
        else:
            raise ValueError("Supported class .... [NOT OK]")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if poisson:
            self.poolscores = PoissonHead(
                in_channel32, modalities, num_classes, kmax=kmax, kmin=kmin,
                alpha=alpha, dropout=dropout, tau=tau)
        else:
            self.poolscores = WildCatClassifierHead(
                in_channel32, modalities, num_classes, kmax=kmax, kmin=kmin,
                alpha=alpha, dropout=dropout)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, seed=None, prngs_cuda=None):
        """
        Forward function.
        :param x: input.
        :param seed: int, a seed for the case of Multigpus to guarantee
        reproducibility for a fixed number of GPUs.
        See  https://discuss.pytorch.org/t/
        did-anyone-succeed-to-reproduce-their-
        code-when-using-multigpus/47079?u=sbelharbi
        In the case of one GPU, the seed in not necessary
        (and it will not be used); so you can set it to None.
        :param prngs_cuda: value returned by torch.cuda.get_prng_state().
        :return:
        """
        # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x = self.relu1(self.bn1(self.conv1(x)))
        # 1 / 2: [n, 64, 240, 240]   --> x2^1 to get back to 1.
        x = self.relu2(self.bn2(self.conv2(x)))
        # 1 / 2: [2, 128, 240, 240]  --> x2^1 to get back to 1.
        x = self.relu3(self.bn3(self.conv3(x)))
        # 1 / 4:  [2, 128, 120, 120] --> x2^2 to get back to 1.
        x = self.maxpool(x)
        # 1 / 4:  [2, 64/256/--, 120, 120]   --> x2^2 to get back to 1.
        x_4 = self.layer1(x)
        # 1 / 8:  [2, 128/512/--, 60, 60]    --> x2^3 to get back to 1.
        x_8 = self.layer2(x_4)
        # 1 / 16: [2, 256/1024/--, 30, 30]   --> x2^4 to get back to 1.
        x_16 = self.layer3(x_8)
        # 1 / 32: [n, 512/2048/--, 15, 15]   --> x2^5 to get back to 1.
        x_32 = self.layer4(x_16)

        # classifier at 32.
        scores32, maps32 = self.poolscores(
            x=x_32, seed=seed, prngs_cuda=prngs_cuda)

        return scores32, maps32

    def get_nb_params(self):
        """
        Count the number of parameters within the model.

        :return: int, number of learnable parameters.
        """
        return sum([p.numel() for p in self.parameters()])

    def __str__(self):
        return "{}(): deep module.".format(
                self.__class__.__name__)


def load_url(url, model_dir='./pretrained', map_location=torch.device('cpu')):
    """
    Download pre-trained models.
    :param url: str, url of the pre-trained model.
    :param model_dir: str, path to the temporary folder where the pre-trained
    models will be saved.
    :param map_location: a function, torch.device, string, or dict specifying
    how to remap storage locations.
    :return: torch.load() output. Loaded dict state.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], namenet="ResNet18", **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], namenet="ResNet50", **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], namenet="ResNet101", **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def test_all_resnet():
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import softmax

    num_classes = 20
    batch = 1
    poisson = True
    tau = 1.
    models = [resnet18(pretrained=False, num_classes=num_classes, dropout=0.5,
                       poisson=poisson, tau=tau),
              resnet50(pretrained=True, num_classes=num_classes, dropout=0.5,
                       poisson=poisson, tau=tau),
              resnet101(pretrained=True, num_classes=num_classes, dropout=0.5,
                        poisson=poisson, tau=tau)]
    for model in models:
        print("====================== START ===========================")
        print("Testing {}".format(model.namenet))
        model.train()
        print("Num. parameters: {}".format(model.get_nb_params()))
        cuda = "1"
        print("cuda:{}".format(cuda))
        print("DEVICE BEFORE: ", torch.cuda.current_device())
        DEVICE = torch.device(
            "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # torch.cuda.set_device(int(cuda))
            pass

        print("DEVICE AFTER: ", torch.cuda.current_device())
        # DEVICE = torch.device("cpu")
        model.to(DEVICE)
        x = torch.randn(batch, 3, 480, 480)
        x = x.to(DEVICE)
        tx = dt.datetime.now()
        scores, maps = model(x)
        print("Forward time took: {}".format(dt.datetime.now() - tx))
        print(x.size(), scores.size(), maps.size())
        print(scores)
        path = os.path.join("../data/debug/net", model.namenet)
        if not os.path.exists(path):
            os.makedirs(path)
        xint = range(0, num_classes)
        for kk in range(batch):
            fig = plt.figure()
            prob = softmax(scores[kk, :].cpu().detach().numpy())
            title = "{} Network scores".format(model.namenet)
            if poisson:
                title = "Hard-wired Poisson distribution in the  output of " \
                        "a randomly initialized {} \n" \
                        "Number of classes: {}".format(
                         model.namenet, num_classes)

            plt.bar(x=np.arange(num_classes),
                    height=prob,
                    align="center",
                    width=0.98, alpha=1., color="blue")
            fig.suptitle(title, fontsize=10)
            plt.xlabel("Labels")
            plt.ylabel("Posterior probabilities")
            plt.xticks(xint)
            fig.savefig(os.path.join(path, "{}-bar.png".format(kk)))
            plt.close(fig)

            fig = plt.figure()
            plt.fill_between(np.arange(num_classes), 0., prob,
                             facecolor="blue", alpha=1.)
            fig.suptitle(title)
            plt.xlabel("Labels")
            plt.ylabel("Posterior probabilities")
            fig.savefig(os.path.join(path, "{}-fill.png".format(kk)))
            plt.close(fig)
        print("====================== END ===========================")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_all_resnet()
