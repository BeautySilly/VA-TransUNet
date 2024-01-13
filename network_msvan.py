import torch
import torch.nn as nn
import math
import warnings
from timm.models.layers import DropPath
from torch.nn.modules.utils import _pair as to_2tuple

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            stride=2,
            padding=1,
            use_batchnorm=False,
            up_flag=False
    ):
        super().__init__()
        self.up_flag = up_flag
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            stride=stride,
            kernel_size=3,
            padding=padding,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        if self.up_flag:
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(BaseModule): # 深度可分离卷积
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        # attn = attn + attn_0 + attn_1 + attn_2
        attn = attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


class MSCAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                # patch_embed = StemConv(embed_dims[i-1], embed_dims[i], norm_cfg=norm_cfg)
                
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)
                
            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:

            super(MSCAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs




class DecoderCup(nn.Module):
    def __init__(self, decoder_channels=(128, 64, 32, 16), encoder_channels=(32, 64, 160, 256)):
        super(DecoderCup, self).__init__()
        self.encoder_channels = encoder_channels[::-1]
        self.decoder_channels = decoder_channels


        pr_enc = decoder_channels[0]
        self.block_list = nn.ModuleList()
        for skip_channel, out_channel, flag in zip(self.encoder_channels, self.decoder_channels, [False, True, True, True]):
            self.block_list.append(ConvBlock(in_channels=pr_enc,
                                             skip_channels=skip_channel,
                                             out_channels=out_channel,
                                             padding=1,
                                             stride=1,
                                             up_flag=flag,
                                             use_batchnorm=True
                                             ))
            pr_enc = out_channel
        self.fin_conv = ConvBlock(in_channels=pr_enc, out_channels=decoder_channels[-1],
                                  stride=1, padding=1, up_flag=True, use_batchnorm=True)
        self.fin_conv2 = Conv2dReLU(in_channels=decoder_channels[-1],
                                    out_channels=decoder_channels[-1], kernel_size=3, padding=1)

    def forward(self, x, encoder_list_inv):

        for layer, y in zip(self.block_list, encoder_list_inv):
            x = layer(x, y)
                    
        
        x = self.fin_conv(x)
        x = self.fin_conv2(x)
        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class SegUNet(nn.Module):
    def __init__(self, img_size, seg_head=9):
        super(SegUNet, self).__init__()

        self.van_encoder = MSCAN(embed_dims=[64, 128, 320, 512], 
                               mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2],
                               drop_rate=0.0, drop_path_rate=0.1)
        
        checkpoint = torch.load("./pretrain_model/segnext_small_512x512_ade_160k.pth")["state_dict"]

        

        net_state_dict = self.van_encoder.state_dict()
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if k.replace("backbone.", "") in net_state_dict}
        
        
        self.van_encoder.load_state_dict(pretrained_dict, strict=False)
        
        self.hidden_cov = Conv2dReLU(in_channels=512, out_channels=128, kernel_size=3, padding=1, stride=1,
                                     use_batchnorm=True)
        
        self.decoder = DecoderCup(decoder_channels=(128, 64, 32, 16), encoder_channels=(64, 128, 320, 512))
        self.seg_head = SegmentationHead(in_channels=16, out_channels=seg_head, kernel_size=3)

    def forward(self, x):
        
        x_list = self.van_encoder(x)
        x = x_list[-1]
        x = self.hidden_cov(x)
        x = self.decoder(x, x_list[::-1])
        logits = self.seg_head(x)

        return logits

from torchsummary import summary

if __name__ == '__main__':

    img_size = (1, 3, 224, 224)
    model = SegUNet(224)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #if torch.cuda.device_count() > 1:
    #    print("Use", torch.cuda.device_count(), 'gpus')
    #    model = nn.DataParallel(model)
    model.to(device)

    with torch.no_grad():

        # summary(model, torch.randn(img_size).to(device),
        #         col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        # x_in = torch.randn(img_size).to(device)
        # out = model(x_in)

        """
        for name, module in model._modules.items():
            print(name, ":", module)
        print("---------------")
        """
        # for name, module in model._modules.items():
        #    print(name)
        
        # Params
        # from ptflops import get_model_complexity_info
        # flops, params = get_model_complexity_info(model, (3,224,224),as_strings=True,print_per_layer_stat=True) 
        # print("%s %s" % (flops,params))
        

        

        
        