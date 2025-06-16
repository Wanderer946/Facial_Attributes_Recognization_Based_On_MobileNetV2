from torch import nn

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by divisor (8)
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# Basic ReLU
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):  # kernel_size = 3 means  default 3x3 convolution
        padding = (kernel_size - 1) // 2  # padding = 1 means keep the output space size unchange
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),  # Convolution
            nn.BatchNorm2d(out_channel),  # Normalization
            nn.ReLU6(inplace=True)  # The activation function, limits the output to a maximum of 6
        )


# Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio  # Control the multiple of dimensionality increase (* expand_ratio)(Default is 6)
        self.use_shortcut = stride == 1 and in_channel == out_channel  # 判断是否使用残差连接（条件是：in和out相同且stride=1）

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))  # kernel_size = 1 means default 1x1 convolution (dimensionality upgrade)
        layers.extend([
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),  # DW convolution
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),  # kernel_size = 1 means default 1x1 convolution (dimensionality downgrade)
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual  # use InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        """
        - t是扩展因子
        - c是输出特征矩阵深度channel
        - n是bottleneck的重复次数
        - s是步距(针对第一层, 其他为1)
        """

        self.stage_indices = [0, 6, 12, 17]
        self.feature_blocks = []


        # === Network Construction ===
        features = []
        features.append(ConvBNReLU(3, input_channel, stride=2))

        layer_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            # n repeated blocks
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                layer_idx += 1
        features.append(ConvBNReLU(input_channel, last_channel, 1))  # append a 1x1 conv
        self.features = nn.Sequential(*features)


        # === Classifier Structure ===
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Compress the feature map to a 1×1 space, and the output becomes (B, C, 1, 1).
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Feature Map(B, 1280, 1, 1) → (B, 1280)
            nn.Dropout(0.4),  # During training, the output of certain neurons is randomly set to 0 with a 40% probability
            nn.Linear(last_channel, 256),  # 降维提取抽象特征
            nn.ReLU(),  # 激活函数, 增加非线性表示能力, 防止网络退化为线性模型
            nn.Dropout(0.3),  # Ouput randomly set to 0 with a 30% probability again
            nn.Linear(256, num_classes)
        )

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
                nn.init.zeros_(m.bias)

        """
        Input (224x224x3)
        ↓ ConvBNReLU(32, stride=2)
        ↓ Stage 1: 1x InvertedResidual(32→16)
        ↓ Stage 2: 2x InvertedResidual(16→24)
        ↓ Stage 3: 3x InvertedResidual(24→32)
        ↓ Stage 4: 4x InvertedResidual(32→64)
        ↓ Stage 5: 3x InvertedResidual(64→96)
        ↓ Stage 6: 3x InvertedResidual(96→160)
        ↓ Stage 7: 1x InvertedResidual(160→320)
        ↓ ConvBNReLU(1280)
        ↓ AdaptiveAvgPool2d(1,1)
        ↓ Flatten → Dropout → Linear(1280→256) → Dropout → Linear(256→num_classes)
        ↓ Output
        """

    def forward(self, x):
        ori_feat = low_feat = mid_feat = high_feat = None

        for i, layer in enumerate(self.features):
            x = layer(x)  # The value of channel
            if i == 0:             ori_feat = x.clone()
            if x.shape[1] == 24:   low_feat = x.clone()
            if x.shape[1] == 64:   mid_feat = x.clone()
            if x.shape[1] == 160:  high_feat = x.clone()

        x = self.avgpool(x)
        out = self.classifier(x)
        return out, (ori_feat, low_feat, mid_feat, high_feat)
    
# Show the Layer of the MobileNetV2
# model = MobileNetV2()
# for i, layer in enumerate(model.features):
#     print(f"Layer {i}: {layer}")
