import torch.nn as nn
from .resnet import ResNet, BasicBlock
from torch.hub import load_state_dict_from_url
class ResNet_CIFAR(ResNet):
    def __init__(self, block, layers):
        super(ResNet_CIFAR, self).__init__(block, layers)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
def resnet_cifar(pretrained=False, progress=True, num_classes=1000):
    model = ResNet_CIFAR(BasicBlock, [2, 2, 2, 2])
    if num_classes!=1000:
        model.fc = nn.Linear(512 * model.block.expansion, num_classes)
    return model