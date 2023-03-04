from utils import *
from torch import nn, einsum
from einops.layers.torch import Rearrange
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        from IPython import embed
        embed()
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,16,197,32]*3

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out =out.permute(0,2,1,3).reshape(B,N,-1)
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self,dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self,x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self,img_size,patch_size,dim,depth,heads,mlp_dim,channels=3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0,'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 定义图片切块
        self.to_patch = nn.Sequential(
                                  nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
                                 )
        #定义位置和类别向量
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()


    def forward(self,img):

        x=self.to_patch(img).flatten(2).transpose(1, 2)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.repeat(b,1,1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x=self.dropout(x)
        #运行多个注意力快
        x=self.encoder(x)

        return x.mean(dim = 1)

class Re_Attention(nn.Module):
    def __init__(self,dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Re_Attention, self).__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))
        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        ################和原来Vit保持一致############################
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,16,197,32]*3
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        #############计算Re-attrntion##############################
        # attn=attn.permute(0,3,2,1)@ self.reattn_weights
        # attn = self.reattn_norm(attn).permute(0,3,2,1)
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out)

class Re_Encoder(nn.Module):
    def __init__(self,dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Re_Encoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Re_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self,x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DeepViT(nn.Module):
    def __init__(self, img_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super(DeepViT, self).__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # 定义图片切块
        self.to_patch = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        )
        # 定义位置和类别向量
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = Re_Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

    def forward(self,img):

        x = self.to_patch(img).flatten(2).transpose(1, 2)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # 运行多个注意力快
        x = self.encoder(x)

        return x.mean(dim=1)



