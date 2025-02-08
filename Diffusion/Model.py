import math
import time

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     官方实现的LN是默认对最后一个维度进行的，这里是对channel维度进行的，所以单另设一个类。
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
#         self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise ValueError(f"not support data format '{self.data_format}'")
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             # [batch_size, channels, height, width]
#             mean = x.mean(1, keepdim=True)
#             var = (x - mean).pow(2).mean(1, keepdim=True)
#             x = (x - mean) / torch.sqrt(var + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x
#
#
# class Block(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#     Args:
#         dim (int): Number of input channels.
#         drop_rate (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#
#     def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         # layer scale
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         shortcut = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
#
#         x = shortcut + self.drop_path(x)
#         return x
#
#
# class ConvNeXt(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  -
#           https://arxiv.org/pdf/2201.03545.pdf
#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#
#     def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
#                  dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
#                  head_init_scale: float = 1.):
#         super().__init__()
#         self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
#         # stem为最初的下采样部分
#         stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#                              LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
#         self.downsample_layers.append(stem)
#
#         # 对应stage2-stage4前的3个downsample
#         for i in range(3):
#             downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                                              nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
#             self.downsample_layers.append(downsample_layer)
#
#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         # 构建每个stage中堆叠的block
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
#                   for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#
#         self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
#         self.head = nn.Linear(dims[-1], num_classes)
#         self.apply(self._init_weights)
#         self.head.weight.data.mul_(head_init_scale)
#         self.head.bias.data.mul_(head_init_scale)
#
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.trunc_normal_(m.weight, std=0.2)
#             nn.init.constant_(m.bias, 0)
#
#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#
#         return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
#
#
# def convnext_tiny(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
#     model = ConvNeXt(depths=[3, 3, 9, 3],
#                      dims=[96, 192, 384, 768],
#                      num_classes=num_classes)
#     return model
#
#
# def convnext_small(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[96, 192, 384, 768],
#                      num_classes=num_classes)
#     return model
#
#
# def convnext_base(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
#     # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[128, 256, 512, 1024],
#                      num_classes=num_classes)
#     return model
#
#
# def convnext_large(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
#     # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[192, 384, 768, 1536],
#                      num_classes=num_classes)
#     return model
#
#
# def convnext_xlarge(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[256, 512, 1024, 2048],
#                      num_classes=num_classes)
#     return model

# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#
#
# class TimeEmbedding(nn.Module):
#     def __init__(self, T, d_model, dim):
#         assert d_model % 2 == 0
#         super().__init__()
#         emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
#         emb = torch.exp(-emb)
#         pos = torch.arange(T).float()
#         emb = pos[:, None] * emb[None, :]
#         assert list(emb.shape) == [T, d_model // 2]
#         emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
#         assert list(emb.shape) == [T, d_model // 2, 2]
#         emb = emb.view(T, d_model)
#
#         self.timembedding = nn.Sequential(
#             nn.Embedding.from_pretrained(emb),
#             nn.Linear(d_model, dim),
#             Swish(),
#             nn.Linear(dim, dim),
#         )
#         self.initialize()
#
#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)
#
#     def forward(self, t):
#         emb = self.timembedding(t)
#         return emb
#
#
# class DownSample(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
#         self.initialize()
#
#     def initialize(self):
#         init.xavier_uniform_(self.main.weight)
#         init.zeros_(self.main.bias)
#
#     def forward(self, x, temb):
#         x = self.main(x)
#         return x
#
#
# class UpSample(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
#         self.initialize()
#
#     def initialize(self):
#         init.xavier_uniform_(self.main.weight)
#         init.zeros_(self.main.bias)
#
#     def forward(self, x, temb):
#         _, _, H, W = x.shape
#         x = F.interpolate(
#             x, scale_factor=2, mode='nearest')
#         x = self.main(x)
#         return x
#
#
# class SPPLayer(torch.nn.Module):
#
#     def __init__(self, num_levels, pool_type='max_pool'):
#         super(SPPLayer, self).__init__()
#
#         self.num_levels = num_levels
#         self.pool_type = pool_type
#
#     def forward(self, x):
#         # num:样本数量 c:通道数 h:高 w:宽
#         # num: the number of samples
#         # c: the number of channels
#         # h: height
#         # w: width
#         num, c, h, w = x.size()
#         #         print(x.size())
#         for i in range(self.num_levels):
#             level = i + 1
#             kernel_size = (math.ceil(h / level), math.ceil(w / level))
#             pooling = (
#             math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
#
#             # update input data with padding
#             zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
#             x_new = zero_pad(x)
#
#             # update kernel and stride
#             h_new = 2 * pooling[0] + h
#             w_new = 2 * pooling[1] + w
#
#             kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
#             stride = (math.floor(h_new / level), math.floor(w_new / level))
#
#             # 选择池化方式
#             print(x_new.shape)
#             if self.pool_type == 'max_pool':
#                 try:
#                     tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride)
#                 except Exception as e:
#                     print(str(e))
#                     print(x.size())
#                     print(level)
#             else:
#                 tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride)
#
#             print(tensor.shape)
#             print(i)
#
#             # 展开、拼接
#             if (i == 0):
#                 x_flatten = tensor
#             else:
#                 x_flatten = torch.cat((x_flatten, tensor))
#
#         print(x_flatten)
#         return x_flatten
#
#
# class CSAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(CSAttentionModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
#
#
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = CSAttentionModule(channel)
#
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         return out
#
#
# class AttnBlock(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.group_norm = nn.GroupNorm(32, in_ch)
#         self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.initialize()
#
#     def initialize(self):
#         for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
#             init.xavier_uniform_(module.weight)
#             init.zeros_(module.bias)
#         init.xavier_uniform_(self.proj.weight, gain=1e-5)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         h = self.group_norm(x)
#         q = self.proj_q(h)
#         k = self.proj_k(h)
#         v = self.proj_v(h)
#
#         q = q.permute(0, 2, 3, 1).view(B, H * W, C)
#         k = k.view(B, C, H * W)
#         w = torch.bmm(q, k) * (int(C) ** (-0.5))
#         assert list(w.shape) == [B, H * W, H * W]
#         w = F.softmax(w, dim=-1)
#
#         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
#         h = torch.bmm(w, v)
#         assert list(h.shape) == [B, H * W, C]
#         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
#         h = self.proj(h)
#
#         return x + h
#
#
# class ConvNextBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             nn.GroupNorm(32, in_ch),
#             Swish(),
#             nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
#         )
#         self.temb_proj = nn.Sequential(
#             Swish(),
#             nn.Linear(tdim, out_ch),
#         )
#         self.block2 = nn.Sequential(
#             nn.GroupNorm(32, out_ch),
#             Swish(),
#             nn.Dropout(dropout),
#             nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
#         )
#         if in_ch != out_ch:
#             self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
#         else:
#             self.shortcut = nn.Identity()
#         if attn:
#             self.attn = AttnBlock(out_ch)
#         else:
#             self.attn = nn.Identity()
#         self.initialize()
#
#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)
#         init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
#
#     def forward(self, x, temb):
#         h = self.block1(x)
#         h += self.temb_proj(temb)[:, :, None, None]
#         h = self.block2(h)
#
#         h = h + self.shortcut(x)
#         h = self.attn(h)
#         return h
#
# # # 添加CBAM和ConvNext block 添加方式1
# # class UNet(nn.Module):
# #     def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
# #         super().__init__()
# #         assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
# #         tdim = ch * 4
# #         self.time_embedding = TimeEmbedding(T, ch, tdim)
# #         self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
# #         self.downblocks = nn.ModuleList()
# #
# #         self.cbam1 = CBAM(channel=ch)
# #         self.SPP1 = SPPLayer(num_levels=32)
# #
# #         chs = [ch]  # record output channel when dowmsample for upsample
# #         now_ch = ch
# #         for i, mult in enumerate(ch_mult):
# #             out_ch = ch * mult
# #             for _ in range(num_res_blocks):
# #                 self.downblocks.append(ConvNextBlock(
# #                     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
# #                     dropout=dropout, attn=(i in attn)))
# #                 now_ch = out_ch
# #                 chs.append(now_ch)
# #             if i != len(ch_mult) - 1:
# #                 self.downblocks.append(DownSample(now_ch))
# #                 chs.append(now_ch)
# #
# #         self.middleblocks = nn.ModuleList([
# #             ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=True),
# #             ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=False),
# #         ])
# #
# #         self.upblocks = nn.ModuleList()
# #         for i, mult in reversed(list(enumerate(ch_mult))):
# #             out_ch = ch * mult
# #             for _ in range(num_res_blocks + 1):
# #                 self.upblocks.append(ConvNextBlock(
# #                     in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
# #                     dropout=dropout, attn=(i in attn)))
# #                 now_ch = out_ch
# #             if i != 0:
# #                 self.upblocks.append(UpSample(now_ch))
# #         assert len(chs) == 0
# #
# #         self.tail = nn.Sequential(
# #             nn.GroupNorm(32, now_ch),
# #             Swish(),
# #             nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
# #         )
# #         self.initialize()
# #
# #     def initialize(self):
# #         init.xavier_uniform_(self.head.weight)
# #         init.zeros_(self.head.bias)
# #         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
# #         init.zeros_(self.tail[-1].bias)
# #
# #     def forward(self, x, t):
# #         # Timestep embedding
# #         temb = self.time_embedding(t)
# #
# #         # Downsampling
# #         # x = self.SPP1(x)
# #         # print(x.shape)
# #
# #         h = self.head(x)
# #         h = self.cbam1(h) + h
# #
# #         hs = [h]
# #         for layer in self.downblocks:
# #             h = layer(h, temb)
# #             hs.append(h)
# #         # Middle
# #         for layer in self.middleblocks:
# #             h = layer(h, temb)
# #         # Upsampling
# #         for layer in self.upblocks:
# #             if isinstance(layer, ConvNextBlock):
# #                 h = torch.cat([h, hs.pop()], dim=1)
# #             h = layer(h, temb)
# #
# #         h = self.tail(h)
# #         print(h.shape)
# #         assert len(hs) == 0
# #         return h
#
#
# class UNet(nn.Module):  # # 添加CBAM和ConvNext block 添加方式2
#     def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
#         super().__init__()
#         assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
#         tdim = ch * 4
#         self.time_embedding = TimeEmbedding(T, ch, tdim)
#         self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
#         self.dongconv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, dilation=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, dilation=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, dilation=5),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.fc = nn.Linear(64, 1)
#
#         self.downblocks = nn.ModuleList()
#         # self.cbam1 = CBAM(channel=ch)
#         self.cbam = nn.ModuleList()
#         chs = [ch]  # record output channel when dowmsample for upsample
#         now_ch = ch
#         for i, mult in enumerate(ch_mult):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks):
#                 self.downblocks.append(ConvNextBlock(
#                     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
#                     dropout=dropout, attn=(i in attn)))
#                 self.cbam.append(CBAM(channel=now_ch))
#                 now_ch = out_ch
#                 chs.append(now_ch)
#             if i != len(ch_mult) - 1:
#                 self.downblocks.append(DownSample(now_ch))
#                 chs.append(now_ch)
#
#         self.middleblocks = nn.ModuleList([
#             ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=True),
#             ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=False),
#         ])
#
#         self.upblocks = nn.ModuleList()
#         for i, mult in reversed(list(enumerate(ch_mult))):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks + 1):
#                 self.upblocks.append(ConvNextBlock(
#                     in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
#                     dropout=dropout, attn=(i in attn)))
#                 self.cbam.append(CBAM(channel=now_ch))
#                 now_ch = out_ch
#             if i != 0:
#                 self.upblocks.append(UpSample(now_ch))
#         assert len(chs) == 0
#
#         self.tail = nn.Sequential(
#             nn.GroupNorm(32, now_ch),
#             Swish(),
#             nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
#         )
#         self.initialize()
#
#     def initialize(self):
#         init.xavier_uniform_(self.head.weight)
#         init.zeros_(self.head.bias)
#         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
#         init.zeros_(self.tail[-1].bias)
#
#     def forward(self, x, t):
#         # Timestep embedding
#         temb = self.time_embedding(t)
#         # Downsampling
#         # h = self.dongconv(x)
#         h = self.head(x)
#         # h = self.cbam1(h) + h
#         hs = [h]
#         for layer in self.downblocks:
#             h = layer(h, temb)
#             hs.append(h)
#         # Middle
#         # h = self.dongconv(h)
#         for layer in self.middleblocks:
#             h = layer(h, temb)
#         # Upsampling
#         # h = self.cbam1(h) + h
#         for layer in self.upblocks:
#             if isinstance(layer, ConvNextBlock):
#                 h = torch.cat([h, hs.pop()], dim=1)
#             h = layer(h, temb)
#
#         h = self.tail(h)
#         print(h.shape)
#         assert len(hs) == 0
#         return h
#
#
#
# # # 添加CBAM和ConvNext block和空洞卷积  添加方式1
# # class UNet(nn.Module):
# #     def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
# #         super().__init__()
# #         assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
# #         tdim = ch * 4
# #
# #
# #         self.time_embedding = TimeEmbedding(T, ch, tdim)
# #         self.head = nn.Conv2d(64, ch, kernel_size=3, stride=1, padding=1)
# #         self.dongconv = nn.Sequential(
# #             nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0, dilation=1),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, dilation=2),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, dilation=5),
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(inplace=True),
# #         )
# #         self.fc = nn.Linear(64, 1)
# #
# #         self.downblocks = nn.ModuleList()
# #
# #         self.cbam1 = CBAM(channel=ch)
# #         chs = [ch]  # record output channel when dowmsample for upsample
# #         now_ch = ch
# #         for i, mult in enumerate(ch_mult):
# #             out_ch = ch * mult
# #             for _ in range(num_res_blocks):
# #                 self.downblocks.append(
# #                     ConvNextBlock(
# #                     in_ch=now_ch, out_ch=out_ch, tdim=tdim,
# #                     dropout=dropout, attn=(i in attn)))
# #                 now_ch = out_ch
# #                 chs.append(now_ch)
# #             if i != len(ch_mult) - 1:
# #                 self.downblocks.append(DownSample(now_ch))
# #                 chs.append(now_ch)
# #
# #         self.middleblocks = nn.ModuleList([
# #             ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=True),
# #             ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=False),
# #         ])
# #
# #
# #         self.upblocks = nn.ModuleList()
# #         for i, mult in reversed(list(enumerate(ch_mult))):
# #             out_ch = ch * mult
# #             for _ in range(num_res_blocks + 1):
# #                 self.upblocks.append(ConvNextBlock(
# #                     in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
# #                     dropout=dropout, attn=(i in attn)))
# #                 now_ch = out_ch
# #             if i != 0:
# #                 self.upblocks.append(UpSample(now_ch))
# #         assert len(chs) == 0
# #
# #         self.tail = nn.Sequential(
# #             nn.GroupNorm(32, now_ch),
# #             Swish(),
# #             nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
# #         )
# #         self.initialize()
# #
# #     def initialize(self):
# #         init.xavier_uniform_(self.head.weight)
# #         init.zeros_(self.head.bias)
# #         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
# #         init.zeros_(self.tail[-1].bias)
# #
# #     def forward(self, x, t):
# #         # Timestep embedding
# #         temb = self.time_embedding(t)
# #
# #         # Downsampling
# #         h = self.dongconv(x)
# #         h = self.head(h)
# #         h = self.cbam1(h) + h
# #         hs = [h]
# #         for layer in self.downblocks:
# #             h = layer(h, temb)
# #             hs.append(h)
# #         # Middle
# #         for layer in self.middleblocks:
# #             h = layer(h, temb)
# #         # Upsampling
# #         for layer in self.upblocks:
# #             if isinstance(layer, ConvNextBlock):
# #                 h = torch.cat([h, hs.pop()], dim=1)
# #             h = layer(h, temb)
# #
# #         h = self.tail(h)
# #         print(h.shape)
# #         assert len(hs) == 0
# #         return h
#
# if __name__ == '__main__':
#     batch_size = 16
#     model = UNet(
#         T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
#         num_res_blocks=2, dropout=0.1)
#     x = torch.randn(batch_size, 3, 128, 128)
#     t = torch.randint(1000, (batch_size,))
#     y = model(x, t)
#     print(y.shape)


import math
import time

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from Diffusion.EMABlock import EMA


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EMABlock(nn.Module):
    def __init__(self, channel):
        super(EMABlock, self).__init__()
        self.EMAttetion = EMA(channel)

    def forward(self, x):
        out = self.EMAttetion(x)
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class CSAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CSAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = CSAttentionModule(channel)

    def forward(self, x):
        out = self.channel_attention(x) * x
        return out


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ConvNextBlock(nn.Module): # Convnext
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class kongdong(nn.Module):  #空洞模块
    def __init__(self, in_ch, out_ch):
        super().__init__()
        print(in_ch, out_ch)
        self.dongconv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        h = self.dongconv(x)
        # h = self.fc(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.head = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)         ##灰色改为nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)，彩色是nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1）

        self.downblocks = nn.ModuleList()
        # self.cbam1 = CBAM(channel=ch)
        self.cbam = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        self.kongdong = kongdong(in_ch=now_ch, out_ch=now_ch)
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ConvNextBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                self.cbam.append(EMABlock(channel=now_ch))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ConvNextBlock(now_ch, now_ch, tdim, dropout, attn=False),

        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ConvNextBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                self.cbam.append(EMABlock(channel=now_ch))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)           ##灰色改为nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)，彩色是nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        # h = self.dongconv(x)
        h = self.head(x)
        # h = self.cbam1(h) + h
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        h = self.kongdong(h)
        for layer in self.middleblocks:
            if isinstance(layer, kongdong):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ConvNextBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 64, 64)               ##这个是输出的图片的维度，灰色第一个参数改为1，彩色是3
    t = torch.randint(1000, (batch_size,))
    y = model(x, t)
    print(y.shape)
