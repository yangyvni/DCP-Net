# file: ultralytics/nn/modules/sppf_sfadc.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv_sfadc_lite import Conv_SFADC_Lite

class SPPF_SFADC(nn.Module):
    """SPPF enhanced with FADC-Lite Conv."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv_SFADC_Lite(c1, c_, 1, 1)
        self.cv2 = Conv_SFADC_Lite(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
