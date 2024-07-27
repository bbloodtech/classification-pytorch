from .mobilenetv2 import mobilenetv2
from .mobilenetv3 import mobilenetv3
from .mobileone import mobileone
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_cifar import resnet_cifar
from .resnext import resnext50
from .resnest import resnest50
from .res2net import res2net50
from .vgg import vgg11, vgg13, vgg16, vgg11_bn, vgg13_bn, vgg16_bn
from .vision_transformer import vit_b_16
from .swin_transformer import swin_transformer_base, swin_transformer_small, swin_transformer_tiny
get_model_from_name = {
    "mobilenetv2"               : mobilenetv2,
    "mobilenetv3"               : mobilenetv3,
    "mobileone"                 : mobileone,
    "resnet18"                  : resnet18,
    "resnet34"                  : resnet34,
    "resnet50"                  : resnet50,
    "resnet101"                 : resnet101,
    "resnet152"                 : resnet152,
    "resnet_cifar"              : resnet_cifar,
    "resnext50"                 : resnext50,
    "resnest50"                 : resnest50,
    "res2net50"                 : res2net50,
    "vgg11"                     : vgg11,
    "vgg13"                     : vgg13,
    "vgg16"                     : vgg16,
    "vgg11_bn"                  : vgg11_bn,
    "vgg13_bn"                  : vgg13_bn,
    "vgg16_bn"                  : vgg16_bn,
    "vit_b_16"                  : vit_b_16,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base
}