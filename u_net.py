import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        attn_output, _ = self.mha(x, x, x)
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)
        return attn_output


class UNetWithAttention(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]
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
            self.ups.append(SpatialAttention(feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
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
            x = self.ups[idx + 2](x)

        return self.final_conv(x)


if __name__ == "__main__":
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetWithAttention(in_channels=3, out_channels=1).to(device)
    # 加载图片
    image_path = 'image1.jpg'  # 替换为你的图片路径
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    # 模型推理
    output = model(input_image)
    output = output.cpu().squeeze(0).squeeze(0).detach().numpy()

    # 显示和保存结果
    plt.imshow(output, cmap='gray')
    plt.axis('off')
    plt.savefig('output_image.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()