from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from torch.hub import load_state_dict_from_url
import torch.nn as nn
model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth'
}

class Bottleneck(_Bottleneck):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=32,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 width_per_group=4,
                 base_channels=64):
        super(Bottleneck, self).__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            assert planes % base_channels == 0
        self.mid_channels = groups * width_per_group * planes // base_channels
        self.conv1 = nn.Conv2d(inplanes, self.mid_channels, 1, bias=False)
        self.bn1 = norm_layer(self.mid_channels)
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = norm_layer(self.mid_channels)
        self.conv3 = nn.Conv2d(self.mid_channels, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

class ResNeXt(ResNet):
    def __init__(self,
                 block,
                 layers,
                 groups=32,
                 width_per_group=4):
        super(ResNeXt, self).__init__(block, layers, groups=groups, width_per_group=width_per_group)

def resnext50(pretrained=False, progress=True, num_classes=1000):
    model = ResNeXt(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)
    
    if num_classes!=1000:
        model.fc = nn.Linear(512 * model.block.expansion, num_classes)
    return model


