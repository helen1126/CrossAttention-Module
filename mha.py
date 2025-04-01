import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.tensorboard import SummaryWriter


# 定义多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output, attention_weights


class TextEncoderWithMHA:
    def __init__(self):
        # 加载CLIP文本编码器和分词器
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")

        # 初始化多头注意力层
        d_model = self.text_encoder.config.hidden_size
        num_heads = 8
        self.mha = MultiHeadAttention(d_model, num_heads)

    def encode_text(self, text):
        # 分词
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # 运行CLIP文本编码器
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            last_hidden_state = outputs.last_hidden_state

        # 运行多头注意力层
        attn_output, attention_weights = self.mha(last_hidden_state, last_hidden_state, last_hidden_state)

        # 取[CLS]标记的输出作为语义嵌入
        embeddings = attn_output[:, 0, :]

        # 输出格式
        output = {"embeddings": embeddings, "attention_weights": attention_weights}

        return output


def test_text_encoder():
    encoder = TextEncoderWithMHA()
    text = ["A cat and a dog"]
    result = encoder.encode_text(text)
    print("Test Output:", result)
    # 使用TensorBoard可视化注意力权重
    writer = SummaryWriter('runs/clip_mha')
    attention_weights = result["attention_weights"]
    for i in range(attention_weights.size(1)):
        writer.add_histogram(f'attention_weights/head_{i}', attention_weights[0, i].flatten(), global_step=0)
    writer.close()


if __name__ == "__main__":
    test_text_encoder()
    