import torch
import torch.nn as nn
from IPython import embed

class Block(nn.Module):
    def __init__(self,embed_dim):
        super(Block, self).__init__()
        self.embed_dim=embed_dim
        self.num_heads = 16  # 注意头数量
        self.scale = 0.125  # 公式中的缩放系数

        self.ln1 = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))

        # 注意力模块
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # 前馈传播
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.drop = nn.Dropout(0.6)


    def forward(self,x):
        # 计算注意力
        x1 = self.ln1(x)  # Layer 正则化
        B, N, C = x1.shape
        qkv = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                         4)  # [3,B,12,197,64]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,16,197,32]*3
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 把数据按照比率划分 [0.2000, 0.5000]》》[0.4256, 0.5744]
        x1 = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj(x1)
        x2 = x + x1

        # 前馈传播
        x2 = self.ln1(x2)
        x2 = self.fc1(x2)
        x2 = self.drop(x2)
        x2 = self.gelu(x2)
        x2 = self.fc2(x2)  # torch.Size([8, 197, 768])
        x2 = self.drop(x2)
        x2=x2+x1

        return x2



class Vitransformer(nn.Module):
    def __init__(self,img_size,patch_size,in_channels=3):
        super(Vitransformer, self).__init__()
        self.num_heads=14
        self.scale=0.125
        #计算中间维度
        dim = 3 * patch_size ** 2
        #定义图片切块
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        #计算encoder模块
        self.qkv=nn.Linear(dim,dim*3)
        self.ln1 = nn.LayerNorm(dim)


    def forward(self,x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        #记录注意力
        B,N,C=x.shape
        assert C // self.num_heads!=0.0,f"中间向量维度无法整除注意力头数量"
        qkv=self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,16,197,32]*3
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        embed()


        return qkv


if __name__=='__main__':
    x=torch.rand(1,3,224,224)
    emb=Vitransformer(img_size=224,patch_size=14)

    print(emb(x).shape)