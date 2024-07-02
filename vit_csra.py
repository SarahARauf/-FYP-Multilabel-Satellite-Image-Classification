""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from functools import partial
from .timm_utils import DropPath, to_2tuple, trunc_normal_
from .csra import MHA, CSRA


default_cfgs = {
    'vit_base_patch16_224':  'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_large_patch16_224':'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth'
    }



class Mlp(nn.Module): #Multi-layer perceptron model
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x): #multi-layer percceptron
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module): #Defines the self-attention mechanism
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 64
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv (3, B, 12, N, C/12)
        # q (B, 12, N, C/12)
        # k (B, 12, N, C/12)
        # v (B, 12, N, C/12)
        # attn (B, 12, N, N)
        # x (B, 12, N, C/12)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module): #defines a transformer block comprising self-attention and MLP layers ()

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) #3. normalizing and using the attention

        x = x + self.drop_path(self.mlp(self.norm2(x))) #4. normalizing and then MLP
        return x
    # def forward(self, x):
    #     norm1 = self.norm1(x)
    #     print("norm1",norm1.shape)
    #     attn = self.attn(norm1)
    #     print("mha",attn.shape)
    #     x = x + self.drop_path(attn)
    #     print("residual1", x.shape)

    #     norm2 = self.norm2(x)
    #     print("norm2",norm2.shape)
    #     mlp = self.mlp(norm1)
    #     print("mlp",mlp.shape)
    #     x = x + self.drop_path(mlp)
    #     print("residual2", x.shape)



    #     return x


class PatchEmbed(nn.Module): #defines the patch embedding layer to convert image patches into embeddings
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768): #16*16*3 = 768
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) #images to img patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)#layer to turn an image into patches (in_channels 3)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) #flatten patch feature map
        return x

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     # FIXME look at relaxing size constraints
    #     assert H == self.img_size[0] and W == self.img_size[1], \
    #         f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

    #     conv2 = self.proj(x)
    #     print("conv2d", conv2.shape)

    #     flat = conv2.flatten(2)
    #     print("flatten",flat.shape)
    #     x = self.proj(x).flatten(2).transpose(1, 2) #flatten patch feature map
    #     return x


class HybridEmbed(nn.Module): #defines a hybrid embedding layer that accepts both image patches and feature maps from CNNS
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VIT_CSRA(nn.Module): #defines the hybrid embedding layer that accepts both img patches and feature maps from cnn input
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, cls_num_heads=1, cls_num_cls=80, lam=0.3):
        super().__init__()
        self.add_w = 0.
        self.normalize = False
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            #1. get patch embeddings
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.HW = int(math.sqrt(num_patches))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #2. class token embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) #2. position embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([ #3. transformer encoder
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # We add our MHA (CSRA) beside the orginal VIT structure below
        self.head = nn.Sequential() # delete original classifier
        self.classifier = MHA(input_dim=embed_dim, num_heads=cls_num_heads, num_classes=cls_num_cls, lam=lam)

        self.loss_func = F.binary_cross_entropy_with_logits

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def backbone(self, x):
        B = x.shape[0] #saving batch num

        #print("Before patch embed:", x.shape)
        x = self.patch_embed(x)

        # print("after patch embed:", x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        # print("after cat:", x.shape)

        x = x + self.pos_embed

        # print("after pos embed:", x.shape)

        x = self.pos_drop(x)

        # print("after pos drop:", x.shape)

        i=1
        for blk in self.blocks:
            x = blk(x)
            #print(i, x.shape)
            i=i+1

        # print("after blk:", x.shape)

        x = self.norm(x)

        #print("after norm:", x.shape)


        # (B, 1+HW, C)
        # we use all the feature to form the tensor like B C H W
        x = x[:, 1:]

        # print("after b,1+hw,c:", x.shape)

        b, hw, c = x.shape
        x = x.transpose(1, 2)

        # print("after transpose:", x.shape)
        x = x.reshape(b, c, self.HW, self.HW)
        # print("after reshape:", x.shape)


        return x

    def forward_train(self, x, target):

        # print(x.shape)
        # print("Hello")
        x = self.backbone(x) #gets feature map
        logit = self.classifier(x) #csra classifier
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)
    

        

def _conv_filter(state_dict, patch_size=16): #a utility function to convert patch embedding weights from manual patchify + linear projection to convolution.
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

'''
nitializes the Vision Transformer model with base settings and CSRA 
head for a given input resolution of 224x224 with 16x16 patches.
'''
def VIT_B16_224_CSRA(pretrained=True, cls_num_heads=1, cls_num_cls=80, lam=0.3):
    model = VIT_CSRA(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), cls_num_heads=cls_num_heads, cls_num_cls=cls_num_cls, lam=lam)

    model_url = default_cfgs['vit_base_patch16_224']
    if pretrained:
        state_dict = model_zoo.load_url(model_url)
        model.load_state_dict(state_dict, strict=False)
    return model

'''
Initializes the Vision Transformer model with large settings 
and CSRA head for a given input resolution of 224x224 with 16x16 patches.
pretrained
'''
def VIT_L16_224_CSRA(pretrained=True, cls_num_heads=1, cls_num_cls=80, lam=0.3):
    model = VIT_CSRA(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), cls_num_heads=cls_num_heads, cls_num_cls=cls_num_cls, lam=lam)

    model_url = default_cfgs['vit_large_patch16_224']
    if pretrained:
        state_dict = model_zoo.load_url(model_url)
        model.load_state_dict(state_dict, strict=False)
        # load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
