import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
class RSoftmax(nn.Module):
    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups
    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2)
            x =F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
class SplitAttentionConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 radix=2,
                 reduction_factor=4):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.conv = nn.Conv2d(in_channels,
                            channels * radix,
                            kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups * radix,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.groups)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.groups)
        self.rsoftmax = RSoftmax(radix, groups)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splits = x.view(batch, self.radix, -1, *x.shape[2:])
            gap = splits.sum(dim=1)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn2(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix > 1:
            attens = atten.view(batch, self.radix, -1, *atten.shape[2:])
            out = torch.sum(attens * splits, dim=1)
        else:
            out = atten * x
        return out.contiguous()

class Bottleneck(_Bottleneck):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 width_per_group=4,
                 base_channels=64,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True):
        super(Bottleneck, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups)
        self.groups = groups
        self.width_per_group = width_per_group
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            assert planes % base_channels == 0
        self.mid_channels = (groups * width_per_group * planes // base_channels)
        self.avg_down_stride = avg_down_stride and self.stride > 1
        self.conv1 = nn.Conv2d(inplanes, self.mid_channels, 1, bias=False)
        self.bn1 = norm_layer(self.mid_channels)
        self.conv2 = SplitAttentionConv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1 if self.avg_down_stride else self.stride, padding=dilation, dilation=dilation, groups=groups, radix=radix, reduction_factor=reduction_factor)
        delattr(self, 'bn2')
        self.conv3 = nn.Conv2d(self.mid_channels, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, self.stride, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.avg_down_stride:
            out = self.avd_layer(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class ResNeSt(ResNet):
    def __init__(self,
                 block,
                 layers,
                 groups=1,
                 width_per_group=4,
                 radix=2,
                 reduction_factor=4,
                 avg_down_stride=True):
        super(ResNeSt, self).__init__(block, layers)
        self.groups = groups
        self.width_per_group = width_per_group
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avg_down_stride = avg_down_stride

def resnest50(pretrained=False, progress=True, num_classes=1000):
    model = ResNeSt(Bottleneck, [3, 4, 6, 3])
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'], model_dir='./model_data',
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    
    if num_classes!=1000:
        model.fc = nn.Linear(512 * model.block.expansion, num_classes)
    return model