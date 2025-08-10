import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):  # x[128, 2, 160]
        x1 = self.fn(x, **kwargs)[:, :1, :] + x[:, :1, :]
        x2 = torch.cat((x1, x[:, 1:2, :]), dim=1)
        return x2


class Residualy(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):  # x[128, 2, 160]
        x1 = self.fn(x, **kwargs)
        x1 = x1 + x
        return x1


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):  # dim=40
        super().__init__()
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 0.125  （x**y）表示 x 的 y 次幂

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):  # x[128, 2, 160]
        b, n, _, h = *x.shape, self.heads  # 180 2 4
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对张量进行分块chunk to_qkv[128, 2, 160*3] ->qkv3个[128, 2, 160]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv)  # qkv3个[128, 2, 160] -> qkv3个[128, 4, 2, 40]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # einsum爱因斯坦求和约定

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):  # mask=None
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # nn.ModuleList:任意 nn.Module 的子类加到这个 list 里面
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residualy(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:  # x[128, 2, 160]
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class fNIRS_STCT(nn.Module):

    def __init__(self, n_class, dim, depth, heads, mlp_dim, dim_head=40, emb_dropout=0.):#dim_head=40
        super().__init__()  # [128, 2, 258, 20]
        num_channels = 100
        dropout = 0

        self.chang_shape = nn.Sequential(  # [128, 2, 258, 20]
            Rearrange('b c h w  -> b c w h')  # [128, 2, 20, 258]
        )

        self.channel_attention = nn.Sequential(  # [128, 2, 258, 20]
            Rearrange('b c h w  -> b c w 1 h'),  # [128, 2, 20, 1, 258]
            nn.AdaptiveAvgPool2d((1, 1)),  # [128, 2, 20, 1, 1]
            Rearrange('b c h w a -> b c (h w a)'),  # [128, 2, 20]
            nn.Linear(20, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.Sigmoid())

        self.chang_shape2 = nn.Sequential(  # [128, 2, 258, 20]
            Rearrange('b c h w  -> b c w h')  # [128, 2, 20, 258]
        )

        self.to_time_embedding = nn.Sequential(  # [128, 2, ]
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(258, 1), stride=(258, 1)),  # [128, 8, 1, 20]
            Rearrange('b c h w  -> b h (c w)'),  # [128, 1, 160]
            nn.Linear(160, dim),  # [128, 1, 160]
            nn.LayerNorm(dim))

        self.to_time_embedding2 = nn.Sequential(  # [128, 2, ]
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(129, 1), stride=(129, 1)),  # [128, 8, 1, 20]
            Rearrange('b c h w  -> b h (c w)'),  # [128, 1, 160]
            nn.Linear(160, dim),  # [128, 1, 160]
            nn.LayerNorm(dim))

        self.cnn = nn.Sequential(  # [128, 2, 258, 20]
            nn.Conv2d(in_channels=2, out_channels=9, kernel_size=(258, 20), stride=(258, 20)),  # [128, 24, 1, 1]
            Rearrange('b c h w  -> b h (c w)'),  # [128, 1, 24]
            nn.Linear(9, 9),  # [128, 1, 160]
            nn.LayerNorm(9))

        self.to_channel_embedding = nn.Sequential(  # [128, 2, 258, 20]
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(6, 20), stride=(6, 20)),  # [128, 8, 258, 1]
            Rearrange('b c h w  -> b w (c h)'),  # [128, 1, 8* 43]
            nn.Linear(8 * 43, dim),  # [128, 1, 160]
            nn.LayerNorm(dim))

        self.pos_embedding_time = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_time = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding_channel = nn.Parameter(torch.randn(1, num_channels + 1, dim))
        self.cls_token_channel = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer_time = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer_channel = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.LayerNorm = nn.LayerNorm(dim)
        self.to_latent = nn.Identity()  # identity表示该字段的值会自动更新，不需要我们维护

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class))

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, n_class))

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(dim*2),
            nn.Linear(dim*2, n_class))

        self.mlp_head_channel = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_class))

        self.a = torch.nn.Parameter(torch.randn(1))

        self.b = torch.nn.Parameter(torch.randn(1))

        self.c = torch.nn.Parameter(torch.randn(1))

        self.d = torch.nn.Parameter(torch.randn(1))

        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, img, mask=None):

        x1 = self.to_time_embedding(img.squeeze())  # [128, 1, 160]

        k = torch.sigmoid(x1)
        x1 = x1 + k * x1

        b, n, _ = x1.shape

        cls_tokens = repeat(self.cls_token_time, '() n d -> b n d', b=b)
        x4 = torch.cat((cls_tokens, x1), dim=1)

        b, n, _ = x4.shape

        x4 += self.pos_embedding_time[:, :n]  # [128, 2, 160]
        x4 = self.dropout(x4)
        x4 = self.transformer_time(x4, mask)  # [128, 2, 160]

        x2 = self.to_channel_embedding(img.squeeze())  # [128, 1, 160]

        b, n, _ = x2.shape

        cls_tokens = repeat(self.cls_token_channel, '() n d -> b n d', b=b)
        x3 = torch.cat((cls_tokens, x2), dim=1)

        b, n, _ = x3.shape

        x3 += self.pos_embedding_channel[:, :n]  # [128, 2, 160]
        x3 = self.dropout(x3)
        x3 = self.transformer_channel(x3, mask)  # [128, 2, 160]

        x3 = x3[:, 0]  # [128, 128]
        x3 = torch.cat((x3, x2.squeeze()), dim=1)
        x3 = self.to_latent(x3)  # [128, 128]
        x3 = self.mlp_head2(x3)
        j = torch.tanh(x3) * torch.sigmoid(x3) * torch.sigmoid(x3)

        x4 = x4[:, 0]  # [128, 128]

        x4 = torch.cat((x4, x1.squeeze()), dim=1)


        x4 = self.to_latent(x4)  # [128, 128]

        x4 = self.mlp_head1(x4)
        x4 = x4 + j * self.a

        return x4  # [128, 3]