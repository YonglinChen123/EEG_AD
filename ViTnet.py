import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2
import time
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

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
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    import matplotlib.image as img
    for i in range(width * height):
        ax = plt.subplot(height, width, i + 1)
        plt.axis('off')

        #img = x[0, i, :, :]
        import matplotlib.image as img
        #img = x[0, i, :, :]
        img = x
        # ix = np.unravel_index(i, ax)
        # img = x[ix]

        pmin = np.min(img)
        pmax = np.max(img)
        #img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        #plt.matshow(img[:, :, ::-1], cmap=plt.get_cmap('gray'))
        #plt.matshow(img((15, 15)), cmap='viridis')
        #plt.imshow(img)
        #cm1 = plt.cm.get_cmap('jet')
        #img = img.imread('1003.jpg')
        # plt.colorbar()
        # print(set(img.flatten())) # {0.007843138, 0.011764706, 0.003921569} 和{1，2，3}
        plt.imshow(img)
        #plt.colorbar()
        #plt.matshow(img)
        #plt.colorbar()
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        #image_height = 95
        #image_width = 79
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            # print(2)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes),
        #    # print(3)
        # )

        self.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.Dropout(0.5),
        nn.ReLU(True),
        nn.Linear(512, 3),
        )
    def forward(self, img):
        x = self.to_patch_embedding(img)
        #savepath = r'features_whitegirl'
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv1.png".format(savepath))

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv2.png".format(savepath))
        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv3.png".format(savepath))
        x = self.transformer(x)
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv4.png".format(savepath))

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #draw_features(1, 1, x.detach().cpu().numpy(), "{}/f1_conv5.png".format(savepath))

        x = self.to_latent(x)
        x = self.fc(x)
        return x