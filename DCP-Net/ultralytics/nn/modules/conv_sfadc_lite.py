import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用你代码中的 LiteSEAttention
class LiteSEAttention(nn.Module):
    def __init__(self, c, reduction=0.0625):
        super().__init__()
        hidden = max(int(c * reduction), 16)
        self.fc1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, c, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.act(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w
    
class Conv_SFADC_Lite(nn.Module): # 实际论文中的方法：边缘检测
    """
    Frequency-Adaptive Dilated Convolution (Lite version, enhanced with Laplacian proxy and OmniAttention)
    Suitable for YOLO11s backbone/head.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1,
                 act=True, reduction=0.0625, dilations=(1, 2),
                 proxy_beta=12.0):
        super().__init__()
        # Lite OmniAttention
        self.att = LiteSEAttention(c1, reduction=reduction)

        # Channel attention (AdaKern-lite)
        hidden_dim = max(int(c1 * reduction), 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c1, hidden_dim, 1, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_dim, c2, 1)
        self.sigmoid = nn.Sigmoid()

        # Dual dilated conv branches
        self.conv = nn.Conv2d(c1, c2, k, s,
                               padding=self._auto_pad(k, dilations[0]),
                               groups=g, dilation=dilations[0], bias=False)
        self.conv2 = nn.Conv2d(c1, c2, k, s,
                               padding=self._auto_pad(k, dilations[1]),
                               groups=g, dilation=dilations[1], bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act2 = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Laplacian-based frequency proxy
        lap_kernel = torch.tensor([[0., 1., 0.],
                                   [1., -4., 1.],
                                   [0., 1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('lap_kernel', lap_kernel)
        self.blur = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            self.blur.weight[:] = torch.ones_like(self.blur.weight) / 9.0
        self.beta = proxy_beta

    @staticmethod
    def _auto_pad(k, d):
        return d * (k - 1) // 2

    def _highfreq_proxy(self, x):
        """Compute Laplacian-based high-frequency response."""
        gray = x.mean(1, keepdim=True)
        lap = F.conv2d(gray, self.lap_kernel, padding=1)
        mag = torch.abs(lap)
        mag = self.blur(mag) # 
        mean = mag.mean(dim=[2, 3], keepdim=True)
        return torch.sigmoid(self.beta * (mag - mean))  # [B,1,H,W]


    def forward(self, x):
        # Apply lightweight SE attention
        x = self.att(x)
        # 1️⃣ 通道注意力 (AdaKern-lite)
        w = self.pool(x)
        w = self.act1(self.fc1(w))
        w = self.sigmoid(self.fc2(w))  # [B, C_out, 1, 1]

        # 2️⃣ 高频空间代理 (FreqProxy)
        hf = self._highfreq_proxy(x)

        # 3️⃣ 两个膨胀卷积分支
        y1 = self.conv(x)
        y2 = self.conv2(x)

        # ✅ 核心修复：对齐频率图尺寸
        if hf.shape[-2:] != y1.shape[-2:]:
            hf = F.adaptive_avg_pool2d(hf, output_size=y1.shape[-2:])

        # 4️⃣ 门控融合
        spatial_weight = 0.5 + 0.5 * hf
        lambda_high = w * spatial_weight
        lambda_low = (1 - w) * (1.5 - spatial_weight)

        # 5️⃣ 广播加权输出
        lambda_high = lambda_high.expand_as(y1)
        lambda_low = lambda_low.expand_as(y1)
        # print(f"!!!!!!lambda_high: {lambda_high}, lambda_low: {lambda_low}, y1 mean: {y1}, y2 mean: {y2}")
        out = lambda_high * y1 + lambda_low * y2

        out = self.act2(self.bn(out))
        return out
