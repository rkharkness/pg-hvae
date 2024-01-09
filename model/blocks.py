#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from torch import nn
import torch
import numpy as np
import torch.nn.functional

import sys
sys.path.append("..")

from model.utils import SEBlock3D

class Upsample(nn.Module):
    def __init__(self, n_channels, n_out, size=None, scale_factor=None,  mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size
        self.n_out = n_out
        self.conv_1 = nn.Conv3d(n_channels, n_out, 3, stride=1, padding='same', bias=False, dilation=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)
        return self.conv_1(x)

class BlockEncoder(nn.Module):
    """BN + Swish + Conv (3x3x3) + BN + Swish + Conv (3x3x3) + SE"""

    def __init__(self, in_channels, n_channels, residual=True, with_se=True):
        super(BlockEncoder, self).__init__()
        
        self.residual = residual
        self.with_se = with_se
        

        self.conv0 = nn.Conv3d(in_channels, n_channels, 3, stride=1, padding='same', bias=True, dilation=1)
        self.norm0 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.act0 = nn.SiLU()

        self.conv1 = nn.Conv3d(n_channels, n_channels, 3, stride=1, padding='same', bias=True, dilation=1)
        self.norm1 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.act1 = nn.SiLU()

        self.se = SEBlock3D(n_channels, reduction_ratio=4)
        self.output_channels = n_channels
        
        self.linear = nn.Linear(in_features=n_channels, out_features=n_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W, D)
        """
        out = self.norm0(x)
        out = self.act0(out)
        out = self.conv0(out)

        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv1(out)

        if self.with_se:
            out = self.se(out)
        if self.residual:
            return out + x
        else:
            return out
        

class BlockDecoder(nn.Module):
    def __init__(self, n_channels, ex=6, residual=True, with_se=True):
        super(BlockDecoder, self).__init__()

        self.residual = residual
        self.with_se = with_se

        hidden_dim = int(round(n_channels * ex))
        
        self.norm0 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.conv0 = nn.Conv3d(n_channels, hidden_dim, 3, stride=1, padding='same', bias=True, dilation=1)

        self.norm1 = nn.InstanceNorm3d(hidden_dim, eps=1e-5, momentum=0.05)
        self.act1 = nn.SiLU()

        self.dw_conv1 = nn.Conv3d(hidden_dim, hidden_dim, 5, stride=1, padding='same', bias=True, dilation=1, groups=hidden_dim)
        
        self.norm2 = nn.InstanceNorm3d(hidden_dim, eps=1e-5, momentum=0.05)
        self.act2 = nn.SiLU()
        
        self.se2 = SEBlock3D(hidden_dim, reduction_ratio=4)

        self.conv2 = nn.Conv3d(hidden_dim, n_channels, 1, stride=1, padding='same', bias=True, dilation=1)

        self.output_channels = n_channels

    def forward(self, x):
        out = self.norm0(x)
        out = self.conv0(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dw_conv1(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        if self.with_se:
            out = self.se2(out)

        out = self.conv2(out)
        if self.residual:
            return out + x
        else:
            return out



class BlockFinal(nn.Module):
    def __init__(self, n_channels, n_out, residual=True, with_se=True, stride=1, mean_final=False):
        super(BlockFinal, self).__init__()

        self.residual = residual
        self.with_se = with_se
        self.stride = stride
        self.mean_final = mean_final

        self.bn_0 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.act_0 = nn.SiLU()
        self.conv_0 = nn.Conv3d(n_channels, n_channels, 3, stride=stride, padding=1, bias=True, dilation=1)

        self.bn_1 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv3d(n_channels, n_channels, 3, stride=stride, padding=1, bias=True, dilation=1)
        self.se = SEBlock3D(n_channels, reduction_ratio=4)

        self.bn_2 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.act_2 = nn.SiLU()
        self.conv_2 = nn.Conv3d(n_channels, n_out, 1, stride=1, padding=0, bias=True, dilation=1)
        

    def forward(self, x):
        out = self.bn_0(x)
        out = self.act_0(out)
        out = self.conv_0(out)

        out = self.bn_1(out)
        out = self.act_1(out)
        out = self.conv_1(out)

        if self.with_se:
            out = self.se(out)
        if self.residual:
            out = out + x

        out = self.bn_2(out)
        out = self.act_2(out)
        out = self.conv_2(out)
        if self.mean_final:
            out = torch.mean(out, [2,3,4], keepdim=True)

        return out 


class BlockQ(nn.Module):
    def __init__(self, n_channels, n_hidden, n_out, **kwargs):
        super().__init__()

        self.bn_0 = nn.InstanceNorm3d(n_channels, eps=1e-5, momentum=0.05)
        self.act_0 = nn.SiLU()
        self.conv_0 = nn.Conv3d(n_channels, n_hidden, 3, stride=1, padding='same', bias=True, dilation=1)
        self.bn_1 = nn.InstanceNorm3d(n_hidden, eps=1e-5, momentum=0.05)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv3d(n_hidden, n_hidden, 3, stride=1, padding='same', bias=True, dilation=1)

        self.se = SEBlock3D(n_hidden, reduction_ratio=4)
        self.last_conv = nn.Conv3d(n_hidden, n_out, 1, 1, 0, 1, 1, False)

    def forward(self, x):
        
        out = self.bn_0(x)
        out = self.act_0(out)
        out = self.conv_0(out)
        
        out = self.bn_1(out)
        out = self.act_1(out)
        out = self.conv_1(out)
    
        out = self.se(out)
        
        x = self.last_conv(out)
        return x


class AsFeatureMap_up(nn.Module):
    def __init__(self, input_dim, target_shape, weightnorm=True, **kwargs):
        super().__init__()

        self._input_dim = input_dim
        out_features = np.prod(target_shape)
        self.linear = nn.Linear(input_dim, out_features)
        self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")
        self._output_shp = target_shape

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.linear(x)
        return x.view([batch_size,]+self._output_shp)


class AsFeatureMap_down(nn.Module):
    def __init__(self, input_shape, target_dim, weightnorm=True, **kwargs):
        super().__init__()
        """Convert a feature map to a vector. Normed fully connected layer."""

        self._input_shp = input_shape
        input_features = np.prod(input_shape)
        self.linear = nn.Linear(input_features, target_dim)
        self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x
    
class ResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type: str="replicate", norm_layer: bool=True, use_dropout: bool=True, use_bias: bool=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []

        # Convolutional layer with optional bias
        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding="same", padding_mode=padding_type, bias=use_bias),
            nn.ReLU(True)
        ]
        
        # Optional dropout layer
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # Another convolutional layer with optional bias
        conv_block += [
            nn.Conv3d(dim, dim, kernel_size=3, padding="same", padding_mode=padding_type, bias=use_bias)
        ]

        if norm_layer:
            conv_block += [nn.InstanceNorm3d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # Add skip connections
        return out



class BlockFinalImg(nn.Module):
    def __init__(self, n_channels: int=3, n_out: int=64, last_act: str='tanh') -> None:
        super().__init__()

        self.res = nn.Sequential(
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
            ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
        )
    
        self.convt2 = nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=7, padding="same", padding_mode='reflect')
        self.norm2 = nn.InstanceNorm3d(n_channels)
        self.act = nn.LeakyReLU(0.2, True)
        self.convt3 = nn.Conv3d(in_channels=n_channels, out_channels=n_out, kernel_size=7, padding="same", padding_mode='reflect')
        
        if last_act=='tanh':
            self.act_last = nn.Tanh()
        else:
            self.act_last = nn.Identity()
    
    def forward(self, x):
        out = self.res(x)

        out = self.convt2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.convt3(out)
        out = self.act_last(out)
        return out
    
# class ResnetBlock(nn.Module):
#     def __init__(self, dim: int, padding_type: str="replicate", norm_layer: bool=True, use_dropout: bool=True, use_bias: bool=True):
#         super(ResnetBlock, self).__init__()
        
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         conv_block = []
        
#         conv_block += [
#             nn.Conv3d(dim, dim, kernel_size=3, padding="same", padding_mode=padding_type, bias=use_bias),
#             nn.ReLU(True)
#         ]

#         # Optional dropout layer
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]

#         # Another convolutional layer with optional bias
#         conv_block += [
#             nn.Conv3d(dim, dim, kernel_size=3, padding="same", padding_mode=padding_type, bias=use_bias)
#         ]

#         if norm_layer:
#             conv_block += [nn.InstanceNorm3d(dim)]
#         return nn.Sequential(*conv_block)
    
#     def forward(self, x):
#         out = x + self.conv_block(x)  # Add skip connection
#         return out



# class BlockFinalImg(nn.Module):
#     def __init__(self, n_channels: int=3, n_out: int=64, last_act: str='tanh') -> None:
#         super().__init__()

#         self.res = nn.Sequential(
#             ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
#             ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
#             ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
#             ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
#             ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
#             ResnetBlock(n_channels, padding_type='reflect', norm_layer=nn.InstanceNorm3d, use_dropout=False, use_bias=True),
#         )
#         # print(self.res, 'res')

#         self.convt2 = nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=7, padding="same", padding_mode='reflect')
#         self.norm2 = nn.InstanceNorm3d(n_channels)
#         self.act = nn.LeakyReLU(0.2, True)
#         self.convt3 = nn.Conv3d(in_channels=n_channels, out_channels=n_out, kernel_size=7, padding="same", padding_mode='reflect')

#         if last_act=='tanh':
#             self.act_last = nn.Tanh()
#         else:
#             self.act_last = nn.Identity()
    
#     def forward(self, x):
#         print(x.shape,'in bfi')
#         out = self.res(x)

#         out = self.convt2(out)
#         out = self.norm2(out)
#         out = self.act(out)
#         out = self.convt3(out)
#         out = self.act_last(out)
#         return out


if __name__ == "__main__":
    # Test BlockEncoder
    x = torch.randn(1,1,64,64,64)

    block_encoder = BlockEncoder(1, 12)
    out = block_encoder(x)
    print(f"Encoder block output shape - {out.shape}")

    # Test BlockDecoder
    block_decoder = BlockDecoder(12, 12)
    out = block_decoder(out)
    print(f"Decoder block output shape - {out.shape}")
    
    # Test ResnetBlock
    res_block = ResnetBlock(12)
    out = res_block(out)
    print(f"Resnet block output shape - {out.shape}")

    # Test BlockFinal
    block_final = BlockFinal(12,3)
    out = block_final(out)
    print(f"Final block output shape - {out.shape}")
    
    block_final_img = BlockFinalImg()
    out = block_final_img(out)
    print(f"Final block img output shape - {out.shape}")


