import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sr3_modules.se import ChannelSpatialSELayer


def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:  # 512 x 512
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:  # 1024 x 1024
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:  # 1024 x 1024
            in_channels += inner_channel * channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x))   # (-1, 1)


class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + self.conv1(x1)
        return out


class Fusion_Head(nn.Module):
    def __init__(self, out_channels=3, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=22):
        super(Fusion_Head, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img_size = img_size
        self.time_steps = time_steps

        # Define decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(Block(dim=64, dim_out=64))
        self.decoder.append(AttentionBlock(dim=64, dim_out=64))
        # self.decoder.append(Block(dim=64, dim_out=64))
        # self.decoder.append(AttentionBlock(dim=64, dim_out=64))
        
        self.head_middle = ResBlock(64, 32)
        
        # Final head
        self.rgb_decode2 = HeadLeakyRelu2d(32, 16)
        self.rgb_decode1 = HeadTanh2d(16, out_channels)

    def forward(self, feats):
        x = feats
        lvl = 0
        for layer in self.decoder:
            if isinstance(layer, Block):
                f_s = x
                f_s = layer(f_s)
                if lvl != 0:
                    f_s = f_s + x
                lvl += 1
            else:
                f_s = layer(f_s)
                x = F.interpolate(f_s, scale_factor=2, mode="bilinear", align_corners=True)
        # decoder.shape:  torch.Size([16, 64, 1024, 1024])
        x = self.head_middle(x)
        x = self.downsample(x)
        x = self.rgb_decode2(x)
        x = self.downsample(x)
        rgb_img = self.rgb_decode1(x)
        return rgb_img



# class Fusion_Head(nn.Module):
#     '''
#     Change detection head (version 2).
#     '''

#     def __init__(self, feat_scales, out_channels=3, inner_channel=None, channel_multiplier=None, img_size=256,
#                  time_steps=None):
#         super(Fusion_Head, self).__init__()

#         # Define the parameters of the change detection head
#         feat_scales.sort(reverse=True)
#         self.feat_scales = feat_scales
#         self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
#         self.img_size = img_size
#         self.time_steps = time_steps

#         # Convolutional layers before parsing to difference head
#         self.decoder = nn.ModuleList()
#         for i in range(0, len(self.feat_scales)):
#             dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

#             self.decoder.append(
#                 Block(dim=dim, dim_out=dim, time_steps=time_steps)
#             )

#             if i != len(self.feat_scales) - 1:
#                 dim_out = get_in_channels([self.feat_scales[i + 1]], inner_channel, channel_multiplier)
#                 self.decoder.append(
#                     AttentionBlock(dim=dim, dim_out=dim_out)
#                 )
#         # Final head
#         self.rgb_decode2 = HeadLeakyRelu2d(128, 64)
#         self.rgb_decode1 = HeadTanh2d(64, 3)

#     def forward(self, feats):
#         # Decoder
#         lvl = 0
#         for layer in self.decoder:
#             if isinstance(layer, Block):
#                 f_s = feats[0][self.feat_scales[lvl]]  # feature stacked
#                 if len(self.time_steps) > 1:
#                    for i in range(1, len(self.time_steps)):
#                       f_s = torch.cat((f_s, feats[i][self.feat_scales[lvl]]), dim=1)
#                    f_s = layer(f_s)
#                 if lvl != 0:
#                     f_s = f_s + x
#                 lvl += 1
#             else:
#                 f_s = layer(f_s)
#                 x = F.interpolate(f_s, scale_factor=2, mode="bilinear", align_corners=True)

#         # Fusion Head
#         x = self.rgb_decode2(x)
#         rgb_img = self.rgb_decode1(x)
#         return rgb_img