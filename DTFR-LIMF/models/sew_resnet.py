import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer

__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d,
                 neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        # 移除SeqToANNContainer对时序的处理，直接用Conv+BN
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = neuron(** kwargs)  # 神经元不带时序维度

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron(**kwargs)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # x: [batch, C, H, W]（无T维度）
        identity = x

        # 单步前向：Conv→BN→神经元（无时序操作）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # SEW连接方式保留 不允许使用就地操作，即 out +=
        out = out + identity

        return out


def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)


class SEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=nn.BatchNorm2d, connect_f=None,  # 移除T参数
                 neuron: callable = None, **kwargs):
        super(SEWResNet, self).__init__()
        self.connect_f = connect_f
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group

        # 首层无时序处理
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = neuron(** kwargs)  # 单步神经元
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 移除SeqToANNContainer

        # 构建网络层（无T参数）
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, neuron=neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], norm_layer=norm_layer, neuron=neuron,** kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], norm_layer=norm_layer, neuron=neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], norm_layer=norm_layer, neuron=neuron,** kwargs)

        # 分类头无时序平均
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.flat = nn.Flatten()
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, norm_layer=nn.BatchNorm2d, neuron: callable = None, **kwargs):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 下采样无时序处理
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                neuron(** kwargs)  # 单步神经元
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            neuron=neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                neuron=neuron,** kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, C, H, W]（单时间步输入）
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)  # 无T维度扩展
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)  # 无T维度，直接展平
        x = self.fc(x)

        return x


# 模型构造函数（移除T参数，适配循环输入）
def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers,** kwargs)
    return model


def sew_resnet18(neuron: callable = None, **kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], neuron=neuron, ** kwargs)


def sew_resnet34(neuron: callable = None, **kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], neuron=neuron, ** kwargs)

