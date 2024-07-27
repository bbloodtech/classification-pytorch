import math
import torch
import torch.nn as nn
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet

class Bottle2neck(_Bottleneck):
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
                 scales=4,
                 stage_type='normal'):
        super(Bottle2neck, self).__init__(inplanes, planes, stride=stride, downsample=downsample)
        assert scales > 1, 'Res2Net degenerates to ResNet when scales=1.'
        mid_channels = planes
        width = int(math.floor(mid_channels * (base_width / base_channels)))
        
        self.conv1 = nn.Conv2d(inplanes, width * scales, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scales)
        if stage_type == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                nn.Conv2d(
                    width,
                    width,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False))
            self.bns.append(nn.BatchNorm2d(width))
        self.conv3 = nn.Conv2d(width * scales, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, 'bn2')
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        sp = self.convs[0](spx[0].contiguous())
        sp = self.relu(self.bns[0](sp))
        out = sp
        for i in range(1, self.scales - 1):
            if self.stage_type == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp.contiguous())
            sp = self.relu(self.bns[i](sp))
            out = torch.cat((out, sp), 1)
        if self.stage_type == 'normal' and self.scales != 1:
            out = torch.cat((out, spx[self.scales - 1]), 1)
        elif self.stage_type == 'stage' and self.scales != 1:
            out = torch.cat((out, self.pool(spx[self.scales - 1])), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class Res2Net(ResNet):
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, avg_down=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    norm_layer(planes * block.expansion)
                )
            else:    
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, stage_type='stage',))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, stride=1, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
def res2net50(pretrained=False, progress=True, num_classes=1000):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3])
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnext50_32x4d'], model_dir='./model_data',
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    
    if num_classes!=1000:
        model.fc = nn.Linear(512 * model.block.expansion, num_classes)
    return model
