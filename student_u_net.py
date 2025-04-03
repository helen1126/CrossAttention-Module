import torch
import torch.nn as nn

# 双卷积模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 空间-语义交叉注意力模块
class SpatialSemanticAttention(nn.Module):
    def __init__(self, in_channels, semantic_dim):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.W_q = nn.Linear(semantic_dim, in_channels)
        self.W_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_o = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, semantic_embedding):
        b, c, h, w = x.shape
        # 计算查询
        Q = self.W_q(semantic_embedding).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        # 计算键和值
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        scores = torch.sum(Q * K, dim=1, keepdim=True) / (self.semantic_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)

        # 计算注意力输出
        attn_output = attention_weights * V
        attn_output = self.W_o(attn_output)

        # 融合原始特征图和注意力输出
        output = x + attn_output

        return output

# 轻量化的 UNet 学生模型
class StudentUNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[32, 64, 128, 256], semantic_dim=768
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下采样部分
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 上采样部分
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            self.ups.append(SpatialSemanticAttention(feature, semantic_dim))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, semantic_embedding):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 3]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
            x = self.ups[idx + 2](x, semantic_embedding)

        return self.final_conv(x)