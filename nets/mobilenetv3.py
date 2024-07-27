from torch import nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
# 指定函数接口可以被外部使用
__all__ = ['MobileNetV3', 'mobilenetv3']

model_urls = {
    'mobilenet_v3_small': "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    'mobilenet_v3_large': "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth"
}
# 往上取整为 8的整数倍
def make_divisible(x, dividible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / dividible_by) * dividible_by)

def conv_bn(in_channel, out_channel, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(in_channel, out_channel, 3, stride, 1, bias=False),
        norm_layer(out_channel),
        act_layer(inplace=True)
    )
def conv_1x1_bn(in_channel, out_channel, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(in_channel, out_channel, 1, 1, 0, bias=False),
        norm_layer(out_channel),
        act_layer(inplace=True)
    )

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        # return x * F.relu6(x + 3., inplace=self.inplace) / 6.
        return x * F.hardtanh(x + 3., 0., 6., inplace=self.inplace) / 6.

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()    
        self.inplace = inplace
    def forward(self, x):
        # return F.relu6(x + 3., inplace=self.inplace) / 6.
        return F.hardtanh(x + 3., 0., 6., inplace=self.inplace) / 6.

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class MobileBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, exp, se=False, act='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and in_channel == out_channel

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if act == 'RE':
            act_layer = nn.ReLU
        elif act == 'HS':
            act_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SElayer = SEModule
        else:
            SElayer = Identity
        
        self.conv = nn.Sequential(
            conv_layer(in_channel, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            act_layer(inplace=True),
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            SElayer(exp),
            act_layer(inplace=True),
            conv_layer(exp, out_channel, 1, 1, 0, bias=False),
            act_layer(out_channel)
        )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            mobile_setting = [
                # k exp c    se     nl   s
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1]
            ]
        elif mode == 'small':
            mobile_setting = [
                # k exp c    se     nl   s
                [3, 16,  16, True,  'RE', 2],
                [3, 72,  24, False, 'RE', 2],
                [3, 88,  24, False, 'RE', 1],
                [5, 96,  40, True,  'HS', 2],
                [5, 240, 40, True,  'HS', 1],
                [5, 240, 40, True,  'HS', 1],
                [5, 120, 48, True,  'HS', 1],
                [5, 144, 48, True,  'HS', 1],
                [5, 288, 96, True,  'HS', 2],
                [5, 576, 96, True,  'HS', 1],
                [5, 576, 96, True,  'HS', 1]
            ]
        else:
            raise NotImplementedError
        
        # first layer
        assert input_size % 32 == 0
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, norm_layer=Hswish)]
        self.classifier = []

        # mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, act_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, act_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            NotImplementedError
        
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, n_class)
        )
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def mobilenetv3(pretrained=False, num_classes=1000, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenetv2'], model_dir='./model_data')
        model.load_state_dict(state_dict, strict=True)
    if num_classes!=1000:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    return model

if __name__ == '__main__':
    import torch
    net = mobilenetv3(num_classes=3)
    print('mobilenetv3:\n', net.features[-1])
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    input_size = (1, 3, 512, 512)
    x = torch.randn(input_size)
    out = net(x)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile
    flops, params = profile(net, input_size=input_size)
    print(flops)
    print(params)
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    out = net(x)