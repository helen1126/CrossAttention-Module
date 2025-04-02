import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

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

# UNet模块
class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]
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

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet with Attention')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='Number of output channels')
    args = parser.parse_args()

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    output = model(input_image)
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
    plt.savefig('output_image1.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()