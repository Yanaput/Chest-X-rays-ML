import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class BasicConvBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, drop_prob=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_c, out_c, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.down  = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.drop_prob = drop_prob

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down is not None:
            identity = self.down(identity)
        out = out + identity
        if self.drop_prob > 0 and self.training:
            # stochastic depth: drop residual with prob p
            if torch.rand(1, device=out.device) < self.drop_prob:
                out = identity
        return F.relu(out, inplace=True)


class CNN(nn.Module):
    """
    Lightweight ResNet-like CNN for 1-channel medical images.
    Input: [B,1,H,W] ; Output: 14 logits for multi-label classification.
    """
    def __init__(self, num_classes=14, in_chans=1, widths=(32, 64, 128, 256), drop_path_rate=0.1):
        super().__init__()
        self.stem = ConvBNAct(in_chans, widths[0], k=7, s=2, p=3)   # /2
        self.pool = nn.MaxPool2d(3, 2, 1)                           # /4
        # 4 stages with downsample at first block of each stage (except stage0)
        dp = [i * (drop_path_rate / (sum([2,2,2,2])-1)) for i in range(sum([2,2,2,2]))]
        it = iter(dp)
        self.stage1 = nn.Sequential(BasicConvBlock(widths[0], widths[0], 1, next(it)),
                                    BasicConvBlock(widths[0], widths[0], 1, next(it)))
        self.stage2 = nn.Sequential(BasicConvBlock(widths[0], widths[1], 2, next(it)),
                                    BasicConvBlock(widths[1], widths[1], 1, next(it)))
        self.stage3 = nn.Sequential(BasicConvBlock(widths[1], widths[2], 2, next(it)),
                                    BasicConvBlock(widths[2], widths[2], 1, next(it)))
        self.stage4 = nn.Sequential(BasicConvBlock(widths[2], widths[3], 2, next(it)),
                                    BasicConvBlock(widths[3], widths[3], 1, next(it)))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(widths[3], num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        # He init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.stem(x))
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        return self.head(x)
