import torch
import torch.nn as nn
import torch.nn.functional as F


class LLaMA(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_len, num_decoders, batch_size):
        super(LLaMA, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.decoders = nn.Sequential(*[Decoder(num_heads, embedding_dim, context_len, batch_size) for _ in range(num_decoders)])
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    
    def forward(self, idx):
        x = self.embeddings(idx)
        x = self.decoders(x)
        x = self.norm(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_len, batch_size):
        super(Decoder, self).__init__()
        self.norm1 = nn.RMSNorm(embedding_dim)
        self.attention = MultiHeadAttention(num_heads, embedding_dim, context_len, batch_size)
        self.norm2 = nn.RMSNorm(embedding_dim)
        self.linear = LinearLayer(embedding_dim, 4 * embedding_dim)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.linear(self.norm2(x))
        return x


class MaskedSelfAttentionWithRoPE(nn.Module):
    def __init__(self, embedding_dim, head_size, context_len, batch_size):
        super(MaskedSelfAttentionWithRoPE, self).__init__()
        self.KeyW = nn.Linear(embedding_dim, head_size, bias=False)
        self.QueryW = nn.Linear(embedding_dim, head_size, bias=False)
        self.ValueW = nn.Linear(embedding_dim, head_size, bias=False)
        self.rope = self.calcualte_rotary_positional_encoding(head_size, context_len, batch_size)

    @staticmethod
    def calcualte_rotary_positional_encoding(head_size, context_len, batch_size):
        rope = torch.zeros((context_len, head_size, head_size))
        for c in range(context_len):
            for k in range(0, head_size, 2):
                i = k
                j = k
                theta = 10000. ** (-2*(i) // head_size)
                rope[c][i][j] = torch.cos(torch.tensor(float(c+1) * theta))
                rope[c][i][j+1] = -torch.sin(torch.tensor(float(c+1) * theta))
                rope[c][i+1][j] = torch.sin(torch.tensor(float(c+1) * theta))
                rope[c][i+1][j+1] = torch.cos(torch.tensor(float(c+1) * theta))
        rope = torch.stack([rope] * batch_size)
        return rope


    def forward(self, x):
        K = self.KeyW(x)
        K = torch.einsum("bimm,bim->bim", self.rope.to(K.device), K)
        Q = self.QueryW(x)
        V = self.ValueW(x)

        score = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5
        mask = torch.tril(torch.ones_like(score))
        #mask = mask.unsqueeze(0).expand(score.size(0), -1, -1)
        score = score.masked_fill(mask == 0, float('-inf'))
        score = F.softmax(score, dim=-1)
        out = score @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, context_len, batch_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.multi_heads = nn.ModuleList([MaskedSelfAttentionWithRoPE(embedding_dim, embedding_dim // num_heads, context_len, batch_size) for _ in range(num_heads)])
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        multi_heads_out = [head(x) for head in self.multi_heads]
        out = torch.cat(multi_heads_out, dim=-1)
        out = self.fc(out)
        out = self.dropout(out)
        return out


class LinearLayer(nn.Module):
    def __init__(self, embedding_dim, feedforward_dim, dropout=0.1):
        super(LinearLayer, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, feedforward_dim)
        self.fcswiglu = nn.Linear(feedforward_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        # swiglu activation for betta = 1.0
        x = F.silu(x) * self.fcswiglu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
