import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from mha import TextEncoderWithMHA

# 假设已经有了教师模型和学生模型
from dynamic_u_net import UNet as DynamicUNet
from student_u_net import StudentUNet

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化教师模型和学生模型
teacher_model = DynamicUNet().to(device)
student_model = StudentUNet().to(device)

# 初始化文本编码器
text_encoder = TextEncoderWithMHA()
text = ["dog","grass"]  # 输入文本
result = text_encoder.encode_text(text)
semantic_embedding = result["embeddings"].to(device)

# 加载图片
image_path = 'image3.jpg'  # 输入图像地址
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
input_image = transform(image).unsqueeze(0).to(device)

# 定义损失函数和优化器
criterion = nn.KLDivLoss(reduction='batchmean')
mse_criterion = nn.MSELoss()
student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 知识蒸馏训练
num_epochs = 1
temperature = 2.0
alpha = 0.5

for epoch in range(num_epochs):
    # 确保每次迭代都重新计算教师模型输出
    teacher_output = teacher_model(input_image, semantic_embedding, [1.0] * len(text)).detach()

    # 确保每次迭代都重新计算学生模型输出
    student_output = student_model(input_image, semantic_embedding)

    # 蒸馏损失
    distillation_loss = criterion(
        torch.log_softmax(student_output / temperature, dim=1),
        torch.softmax(teacher_output / temperature, dim=1)
    )

    # 学生模型的原始损失（这里假设使用简单的均方误差）
    mse_loss = mse_criterion(student_output, input_image)

    # 总损失
    total_loss = alpha * distillation_loss + (1 - alpha) * mse_loss

    # 反向传播和优化
    student_optimizer.zero_grad()
    try:
        total_loss.backward()
    except RuntimeError as e:
        print(f"Error during backward pass: {e}")
        print(f"teacher_output shape: {teacher_output.shape}")
        print(f"student_output shape: {student_output.shape}")
        break
    student_optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}')
