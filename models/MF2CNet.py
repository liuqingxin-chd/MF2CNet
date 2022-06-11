import torch.nn as nn
from models.octconv import *
import torch
from models.MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = Conv_BN_ACT(inplanes, width, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, norm_layer=norm_layer,
                                 alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.conv3 = Conv_BN(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                             alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))
        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None
        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None
        return x_h, x_l


class MF2CNet(nn.Module):
    def __init__(self, block, layers, num_classes=21, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(MF2CNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(4, 4)
        self.avgpool3 = nn.AvgPool2d(2, 2)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, alpha_in=0)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, alpha_out=0, output=True)
        self.conv2_1 = nn.Conv2d(256, 128, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_1_l = nn.Conv2d(256, 128, 1, 1)
        self.bn2_1_l = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(512, 128, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv2_2_l = nn.Conv2d(512, 128, 1, 1)
        self.bn2_2_l = nn.BatchNorm2d(128)
        self.conv2_3 = nn.Conv2d(2048, 128, 1, 1)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.act = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(640)
        self.fc1 = nn.Linear(205120, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, alpha_in=alpha_in, alpha_out=alpha_out)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, alpha_in, alpha_out, norm_layer, output))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha_in=0 if output else 0.5, alpha_out=0 if output else 0.5, output=output))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # s1
        x_h, x_l = self.layer1(x)
        # s2
        x_h, x_l = self.layer2((x_h, x_l))
        x_h2, x_l2 = x_h, x_l
        x_ph2 = self.avgpool2(x_h2)
        x_pl2 = self.avgpool3(x_l2)
        # s3
        x_h, x_l = self.layer3((x_h, x_l))
        x_h3, x_l3 = x_h, x_l
        x_ph3 = self.avgpool3(x_h3)
        # s4
        x_h, x_l = self.layer4((x_h, x_l))
        x_h4 = x_h
        x_1 = self.act(self.bn2_1(self.conv2_1(x_ph2)))
        x_1_l = self.act(self.bn2_1_l(self.conv2_1_l(x_pl2)))
        x_2 = self.act(self.bn2_2(self.conv2_2(x_ph3)))
        x_2_l = self.act(self.bn2_2_l(self.conv2_2_l(x_l3)))
        x_3 = self.act(self.bn2_3(self.conv2_3(x_h4)))
        x = torch.cat([x_1, x_1_l, x_2, x_2_l, x_3], dim=1)
        x = self.bn3(x)
        # covarience pooling
        x = CovpoolLayer(x)
        x = SqrtmLayer(x, 5)
        x = TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def Net(**kwargs):
    model = MF2CNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

net = Net()
x = torch.randn(1, 3, 224, 224)
y = net(x)
print(y.size())
