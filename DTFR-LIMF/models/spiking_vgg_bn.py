import torch.nn as nn
from spikingjelly.clock_driven import layer

__all__ = [
    'SpikingVGGBN', 'spiking_vgg11_bn', 'spiking_vgg13_bn', 'spiking_vgg16_bn', 'spiking_vgg19_bn'
]

cfg = {

    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}

class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        # 获取初始输入的H和W（从kwargs中获取）
        self.init_h = kwargs.get('init_h', 0)  # 默认输入高度，可根据实际情况调整
        self.init_w = kwargs.get('init_w', 0)  # 默认输入宽度，可根据实际情况调整

        # 为每个layer计算并传递对应的H和W
        current_h, current_w = self.init_h, self.init_w
        self.layer1, current_h, current_w = self._make_layers(
            cfg[vgg_name][0], dropout, neuron, current_h, current_w, **kwargs
        )
        self.layer2, current_h, current_w = self._make_layers(
            cfg[vgg_name][1], dropout, neuron, current_h, current_w, **kwargs
        )
        self.layer3, current_h, current_w = self._make_layers(
            cfg[vgg_name][2], dropout, neuron, current_h, current_w, **kwargs
        )
        self.layer4, current_h, current_w = self._make_layers(
            cfg[vgg_name][3], dropout, neuron, current_h, current_w, **kwargs
        )
        self.layer5, current_h, current_w = self._make_layers(
            cfg[vgg_name][4], dropout, neuron, current_h, current_w, **kwargs
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # only dvs
            # nn.Dropout(0.25),
            #
            nn.Linear(512 * 7 * 7, num_classes),
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout, neuron, h, w, **kwargs):
        layers = []
        current_h, current_w = h, w  # 当前特征图的高度和宽度
        for x in cfg:
            if x == 'M':
                # 池化层会改变特征图尺寸 (AvgPool2d with kernel_size=2, stride=2)
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                current_h = current_h // 2  # 池化后高度减半
                current_w = current_w // 2  # 池化后宽度减半
            else:
                # 卷积层 (kernel_size=3, padding=1 不改变特征图尺寸)
                layers.append(nn.Conv2d(
                    self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias
                ))
                layers.append(nn.BatchNorm2d(x))
                # 向神经元层传入当前的H和W
                layers.append(neuron(** kwargs, channel=x, h=current_h, w=current_w))
                layers.append(layer.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers), current_h, current_w

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out

# class SpikingVGGBN(nn.Module):
#     def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
#         super(SpikingVGGBN, self).__init__()
#         self.whether_bias = True
#         self.init_channels = kwargs.get('c_in', 2)
#
#         self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron, **kwargs)
#         self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron, **kwargs)
#         self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron, **kwargs)
#         self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron, **kwargs)
#         self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron, **kwargs)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 7 * 7, num_classes),
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def  _make_layers(self, cfg, dropout, neuron, **kwargs):
#         layers = []
#         for x in cfg:
#             if x == 'M':
#                 layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
#             else:
#                 layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.whether_bias))
#                 layers.append(nn.BatchNorm2d(x))
#                 # kwargs["l_i"] += 1
#                 layers.append(neuron(**kwargs, channel=x))
#                 layers.append(layer.Dropout(dropout))
#                 self.init_channels = x
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.avgpool(out)
#         out = self.classifier(out)
#
#         return out


def spiking_vgg9_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG9', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg11_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg13_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG13', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg16_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG16', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


def spiking_vgg19_bn(neuron: callable = None, num_classes=10, neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG19', neuron=neuron, dropout=neuron_dropout, num_classes=num_classes, **kwargs)


class SpikingRepVGG(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0, num_classes=10, **kwargs):
        super(SpikingRepVGG, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)
        self.neuron = neuron  # 保存神经元类
        self.neuron_kwargs = kwargs  # 神经元参数
        self.dropout = dropout

        # 创建网络层（不含神经元）
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, num_classes),
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        """创建包含三并行分支(conv3, conv1, 恒等映射)的层"""
        layers = []
        for x in cfg:
            if x == 'M':
                # 池化层
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                # 三并行分支结构
                conv3 = nn.Conv2d(
                    self.init_channels, x, kernel_size=3,
                    padding=1, bias=self.whether_bias
                )
                conv1 = nn.Conv2d(
                    self.init_channels, x, kernel_size=1,
                    padding=0, bias=self.whether_bias
                )

                # 恒等映射分支（如果输入输出通道数不同，用1x1卷积调整）
                if self.init_channels == x:
                    identity = nn.Identity()
                else:
                    identity = nn.Conv2d(
                        self.init_channels, x, kernel_size=1,
                        padding=0, bias=self.whether_bias
                    )

                # 批归一化层
                bn = nn.BatchNorm2d(x)

                # 将三个分支和批归一化打包为一个模块
                layers.append(nn.ModuleList([conv3, conv1, identity, bn]))

                # 更新输入通道数
                self.init_channels = x
        return nn.ModuleList(layers)

    def forward(self, x):
        # 处理layer1
        for block in self.layer1:
            if isinstance(block, nn.AvgPool2d):
                x = block(x)
            else:
                conv3, conv1, identity, bn = block
                # 三个分支并行计算
                out3 = conv3(x)
                out1 = conv1(x)
                out_id = identity(x)
                # 合并三个分支的结果
                x = out3 + out1 + out_id
                # 批归一化
                x = bn(x)
                # 应用神经元激活（从layer中提取出来）
                x = self.neuron(**self.neuron_kwargs, channel=x.size(1), h=x.size(2), w=x.size(3))(x)
        x = self.dropout_layers[0](x)

        # 处理layer2
        for block in self.layer2:
            if isinstance(block, nn.AvgPool2d):
                x = block(x)
            else:
                conv3, conv1, identity, bn = block
                out3 = conv3(x)
                out1 = conv1(x)
                out_id = identity(x)
                x = out3 + out1 + out_id
                x = bn(x)
                x = self.neuron(**self.neuron_kwargs, channel=x.size(1), h=x.size(2), w=x.size(3))(x)
        x = self.dropout_layers[1](x)

        # 处理layer3
        for block in self.layer3:
            if isinstance(block, nn.AvgPool2d):
                x = block(x)
            else:
                conv3, conv1, identity, bn = block
                out3 = conv3(x)
                out1 = conv1(x)
                out_id = identity(x)
                x = out3 + out1 + out_id
                x = bn(x)
                x = self.neuron(**self.neuron_kwargs, channel=x.size(1), h=x.size(2), w=x.size(3))(x)
        x = self.dropout_layers[2](x)

        # 处理layer4
        for block in self.layer4:
            if isinstance(block, nn.AvgPool2d):
                x = block(x)
            else:
                conv3, conv1, identity, bn = block
                out3 = conv3(x)
                out1 = conv1(x)
                out_id = identity(x)
                x = out3 + out1 + out_id
                x = bn(x)
                x = self.neuron(**self.neuron_kwargs, channel=x.size(1), h=x.size(2), w=x.size(3))(x)
        x = self.dropout_layers[3](x)

        # 处理layer5
        for block in self.layer5:
            if isinstance(block, nn.AvgPool2d):
                x = block(x)
            else:
                conv3, conv1, identity, bn = block
                out3 = conv3(x)
                out1 = conv1(x)
                out_id = identity(x)
                x = out3 + out1 + out_id
                x = bn(x)
                x = self.neuron(**self.neuron_kwargs, channel=x.size(1), h=x.size(2), w=x.size(3))(x)
        x = self.dropout_layers[4](x)

        # 分类器部分
        x = self.avgpool(x)
        x = self.classifier(x)
        return x