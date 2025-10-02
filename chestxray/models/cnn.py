import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM 
from .gem import GeMPooling

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate
        self.conv1 = ConvBNAct(in_c, out_c, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.drop_rate > 0 and self.training:
            keep_prob = 1 - self.drop_rate
            random_tensor = keep_prob + torch.rand((out.shape[0], 1, 1, 1), device=out.device)
            random_tensor.floor_()  # binarize
            out = out.div(keep_prob) * random_tensor

        out = out + identity
        return F.relu(out, inplace=True)


class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_c, out_c, stride=1, drop_rate=0.0):
        super().__init__()
        self.drop_rate = drop_rate

        self.conv1 = ConvBNAct(in_c, out_c, k=1, s=1, p=0)
        self.conv2 = ConvBNAct(out_c, out_c, k=3, s=stride, p=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_c, out_c * self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_c * self.expansion)
        )

        self.downsample = None
        if stride != 1 or in_c != out_c * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_c * self.expansion)
            )

        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.drop_rate > 0 and self.training:
            keep_prob = 1 - self.drop_rate
            random_tensor = keep_prob + torch.rand((out.shape[0], 1, 1, 1), device=out.device)
            random_tensor.floor_()  # binarize
            out = out.div(keep_prob) * random_tensor

        out += identity
        return self.relu(out)


class CNN(nn.Module):
    """
    CNN model based on ResNet architecture
    Input: [B,1,H,W] ; 
    Output: 14 logits for multi-label classification.
    """
    def __init__(
        self, 
        num_classes=14, 
        in_chans=1, 
        widths=(32 ,64, 128, 256),
        layers = (3, 4, 6, 3),
        block: BottleneckBlock | ResidualBlock = BottleneckBlock,
        drop_path_rate=0.1,

        add_cbam=("stage1", "stage2", "stage3", "stage4"),
        cbam_reduction=8,
        cbam_spatial_kernel=7,
    ):
        super().__init__()
        self.in_channels = in_chans
        self.stem = ConvBNAct(in_chans, widths[0], k=7, s=2, p=3)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.in_channels = widths[0]
        total = sum(layers)
        drop_rates = torch.linspace(0, drop_path_rate, steps=total).tolist()

        dr_idx = 0
        self.stage1 = self._make_layer(
            block, widths[0], layers[0], stride=1, 
            drop_rates=drop_rates[dr_idx:dr_idx+layers[0]]
        )

        dr_idx += layers[0]
        self.stage2 = self._make_layer(
            block, widths[1], layers[1], stride=2, 
            drop_rates=drop_rates[dr_idx:dr_idx+layers[1]]
        )

        dr_idx += layers[1]
        self.stage3 = self._make_layer(
            block, widths[2], layers[2], stride=2, 
            drop_rates=drop_rates[dr_idx:dr_idx+layers[2]]
        )

        dr_idx += layers[2]
        self.stage4 = self._make_layer(
            block, widths[3], layers[3], stride=2, 
            drop_rates=drop_rates[dr_idx:dr_idx+layers[3]]
        )


        self.add_cbam = set(add_cbam)
        self.cbam1 = CBAM(widths[0] * block.expansion, cbam_reduction, cbam_spatial_kernel)
        self.cbam2 = CBAM(widths[1] * block.expansion, cbam_reduction, cbam_spatial_kernel)
        self.cbam3 = CBAM(widths[2] * block.expansion, cbam_reduction, cbam_spatial_kernel)
        self.cbam4 = CBAM(widths[3] * block.expansion, cbam_reduction, cbam_spatial_kernel)  
         
        
        self.head = nn.Sequential(
            GeMPooling(p=3.0, eps=1e-6, learn_p=True),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(widths[-1] * block.expansion, num_classes)
        )
        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def _make_layer(self, block, out_channels, num_blocks, stride, drop_rates=None):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        if drop_rates is None:
            drop_rates = [0.0] * num_blocks

        for i in range(1, num_blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                stride=1,
                drop_rate=drop_rates[i]
            ))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.pool(self.stem(x))
        
        x = self.stage1(x)
        if "stage1" in self.add_cbam: x = self.cbam1(x)

        x = self.stage2(x)
        if "stage2" in self.add_cbam: x = self.cbam2(x)

        x = self.stage3(x)
        if "stage3" in self.add_cbam: x = self.cbam3(x)

        x = self.stage4(x)
        if "stage4" in self.add_cbam: x = self.cbam4(x)

        return self.head(x)
