import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mha import TextEncoderWithMHA
import argparse
import requests
import time
import json
import hashlib

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

class SpatialSemanticAttention(nn.Module):
    def __init__(self, in_channels, semantic_dim):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.W_q = nn.Linear(semantic_dim, in_channels)
        self.W_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_o = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, semantic_embedding, weights):
        b, c, h, w = x.shape
        # 确保 weights 的形状正确
        weights = torch.tensor(weights, device=semantic_embedding.device).unsqueeze(0).unsqueeze(-1)  # 增加批量维度
        # 应用动态权重
        weighted_embedding = semantic_embedding * weights
        # 计算查询
        Q = self.W_q(weighted_embedding).unsqueeze(-1).unsqueeze(-1)
        # 确保 expand 的尺寸和 Q 的维度匹配
        Q = Q.expand(b, -1, -1, h, w)

        # 计算键和值
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力分数
        scores = torch.sum(Q * K.unsqueeze(1), dim=2, keepdim=True) / (self.semantic_dim ** 0.5)

        attention_weights = torch.softmax(scores, dim=-1)

        # 计算注意力输出
        attn_output = attention_weights * V.unsqueeze(1)
        attn_output = attn_output.sum(dim=1)
        attn_output = self.W_o(attn_output)

        # 融合原始特征图和注意力输出
        output = x + attn_output

        return output

# UNet模块
class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], semantic_dim=768
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

    def forward(self, x, semantic_embedding, weights):
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
            x = self.ups[idx + 2](x, semantic_embedding, weights)

        return self.final_conv(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet with Attention')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='Number of output channels')
    args = parser.parse_args()

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化文本编码器
    text_encoder = TextEncoderWithMHA()
    text = ["dog","grass"]  # 输入文本
    result = text_encoder.encode_text(text)
    semantic_embedding = result["embeddings"].to(device)

    # 构造请求数据
    request_data = {
        "base_prompt": "A dog on the grass",
        "attributes": [
            {
                "name": "object",
                "type": "text",
                "value": text[0],
                "initial_weight": 0.8,
                "constraints": {
                    "min_weight": 0.4,
                    "max_weight": 0.9,
                }
            },
            {
                "name": "background",
                "type": "text",
                "value": text[1],
                "initial_weight": 0.2,
                "constraints": {
                    "min_weight": 0.1,
                    "max_weight": 0.5,
                }
            }
        ],
        "temperature": 1.8,
        "fallback_strategy": "creative",
    }

    # 获取时间戳
    timestamp = str(int(time.time()))
    # 检查时间戳是否在有效范围内（假设有效期为 60 秒）
    if int(time.time()) - int(timestamp) > 60:
        print("时间戳已过期，请重新获取")
        timestamp = str(int(time.time()))

    data_str = json.dumps(request_data, sort_keys=True)

    # 计算签名
    message = f"{data_str}{timestamp}"
    signature = hashlib.sha256(message.encode()).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": "api_key",
        "X-Timestamp": timestamp,
        "X-Signature": signature,
    }

    # 调用 dynamic_weights 接口
    url = 'http://127.0.0.1:5000/api/v5/weight/calculate'
    response = requests.post(url, headers=headers, json=request_data)
    if response.status_code == 200:
        response_data = response.json()
        if response_data["code"] == 200:
            final_weights = response_data["data"]["final_weights"]
            weights = [final_weights[key] for key in text]
        else:
            print(f"Failed to get dynamic weights: {response_data}")
            weights = [1.0] * len(text)
    else:
        print(f"Failed to get dynamic weights: {response.text}")
        weights = [1.0] * len(text)

    model = UNet(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    # 加载图片
    image_path = 'image3.jpg'  # 输入图像地址
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    # 模型推理
    output = model(input_image, semantic_embedding, weights)
    output = output.cpu().squeeze(0).detach().numpy()

    # 归一化到 [0, 1] 范围
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # 调整维度以符合 imshow 要求 (H, W, C)
    if args.out_channels == 1:
        output = output.squeeze()
    else:
        output = output.transpose(1, 2, 0)

    # 显示和保存结果
    plt.imshow(output)
    plt.axis('off')
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()