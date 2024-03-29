from torch import nn
import torch
import torch.nn.functional

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv23d) or isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.InstanceNorm2d):
            module.weight = nn.init.constant_(module.weight, 1)
            module.bias = nn.init.constant_(module.bias, 0)



class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock3D, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Squeeze (Global Average Pooling)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # Excitation (Fully connected layers)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        out = self.avg_pool(x).view(b, c)
        
        # Excitation: Fully connected layers
        out = self.fc(out).view(b, c, 1, 1, 1)
        
        # Scale the original feature maps
        out = x * out
        
        return out


class ConsistentOutputShape3DLayer(nn.Module):
    def __init__(self, min_output_shape):
        super(ConsistentOutputShape3DLayer, self).__init__()
        self.min_output_shape = min_output_shape

    def forward(self, x):
        # Get the current shape of the input tensor
        input_shape = x.size()

        # Determine how much padding or trimming is needed
        pad_shape = [max(0, min_shape - curr_shape) for min_shape, curr_shape in zip(self.min_output_shape, input_shape)]
        # trim_shape = [max(0, curr_shape - min_shape) for min_shape, curr_shape in zip(self.min_output_shape, input_shape)]
        
        print(pad_shape, "pad shape")

        # Apply padding if needed
        if any(pad_shape):
            x = nn.functional.pad(x, (0, pad_shape[4], 0, pad_shape[3], 0, pad_shape[2], 0, pad_shape[1], 0, pad_shape[0]))

        # # Apply trimming if needed
        # if any(trim_shape):
        #     x = x[:, :, :trim_shape[1], :trim_shape[2], :trim_shape[3]]

        return x


def soft_clamp(x: torch.Tensor, v: int=10):
    return x.div(v).tanh_().mul(v)

def soft_clamp_img(x: torch.Tensor):
    return (x.div(5).tanh_() + 1 ) / 2 