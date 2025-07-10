import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, neuron, functional

class SpikingDoubleConv(nn.Module):
    """(convolution => [BN] => LIF) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            layer.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(mid_channels),
            neuron.LIFNode(),
            layer.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode()
        )

    def forward(self, x):
        # x shape: [T, N, C, H, W]
        T, N, C, H, W = x.shape
        x_reshaped = x.reshape(T * N, C, H, W)  # Combine T and N dimensions
        out = self.double_conv(x_reshaped)
        return out.reshape(T, N, -1, H, W)  # Restore T dimension

class SpikingSelfAttention(nn.Module):
    """
    A simplified Spiking Self-Attention (SSA) module inspired by Spike2Former.
    This module processes data over T timesteps.
    """
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        if self.head_dim * num_heads != self.in_channels:
            raise ValueError("in_channels must be divisible by num_heads")

        self.q_conv = layer.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k_conv = layer.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v_conv = layer.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.attn_lif = neuron.LIFNode(v_threshold=0.5)
        
        self.out_conv = layer.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.out_bn = layer.BatchNorm2d(in_channels)
        self.out_lif = neuron.LIFNode()

    def forward(self, x):
        # x shape: [T, N, C, H, W]
        T, N, C, H, W = x.shape
        
        # Reshape to combine T and N dimensions for Conv2d operations
        x_reshaped = x.reshape(T * N, C, H, W)
        
        # Apply conv layers
        q = self.q_conv(x_reshaped)
        k = self.k_conv(x_reshaped)
        v = self.v_conv(x_reshaped)
        
        # Restore temporal dimension
        q = q.reshape(T, N, C, H, W)
        k = k.reshape(T, N, C, H, W)
        v = v.reshape(T, N, C, H, W)
        
        # Reshape for multi-head attention
        # [T, N, C, H, W] -> [T, N, num_heads, head_dim, H*W]
        q = q.view(T, N, self.num_heads, self.head_dim, H * W).permute(0, 1, 2, 4, 3) # [T, N, num_heads, H*W, head_dim]
        k = k.view(T, N, self.num_heads, self.head_dim, H * W) # [T, N, num_heads, head_dim, H*W]
        v = v.view(T, N, self.num_heads, self.head_dim, H * W).permute(0, 1, 2, 4, 3) # [T, N, num_heads, H*W, head_dim]

        # Calculate attention scores
        # The matmul is over the last two dimensions (H*W, head_dim) and (head_dim, H*W)
        attn_scores = torch.matmul(q, k) / (self.head_dim ** 0.5) # [T, N, num_heads, H*W, H*W]
        
        # Apply spiking activation to attention scores
        spiking_attn = self.attn_lif(attn_scores)
        
        # Apply attention to values
        # The matmul is over (H*W, H*W) and (H*W, head_dim)
        out = torch.matmul(spiking_attn, v) # [T, N, num_heads, H*W, head_dim]
        
        # Reshape back to image format
        out = out.permute(0, 1, 2, 4, 3).contiguous().view(T, N, C, H, W)
        
        # Final output layers - need to handle temporal dimension
        out_reshaped = out.reshape(T * N, C, H, W)
        out_conv = self.out_conv(out_reshaped)
        out_bn = self.out_bn(out_conv)
        out_final = out_bn.reshape(T, N, C, H, W)
        out_final = self.out_lif(out_final)
        
        return out_final + x # Residual connection


class Down(nn.Module):
    """Downscaling with maxpool then SpikingDoubleConv"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.maxpool = layer.MaxPool2d(2)
        self.conv = SpikingDoubleConv(in_channels, out_channels)
        if use_attention:
            self.attn = SpikingSelfAttention(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        # x shape: [T, N, C, H, W]
        T, N, C, H, W = x.shape
        x_reshaped = x.reshape(T * N, C, H, W)
        x_pooled = self.maxpool(x_reshaped)
        _, _, H_new, W_new = x_pooled.shape
        x = x_pooled.reshape(T, N, C, H_new, W_new)
        x = self.conv(x)
        x = self.attn(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SpikingDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = layer.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SpikingDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 and x2 shapes: [T, N, C, H, W]
        T, N, C1, H1, W1 = x1.shape
        
        if hasattr(self, 'up') and isinstance(self.up, nn.Upsample):
            # Handle nn.Upsample which doesn't understand temporal dimension
            x1_reshaped = x1.view(T * N, C1, H1, W1)
            x1_upsampled = self.up(x1_reshaped)
            _, _, H_up, W_up = x1_upsampled.shape
            x1 = x1_upsampled.view(T, N, C1, H_up, W_up)
        else:
            # Handle ConvTranspose2d with proper reshaping
            x1_reshaped = x1.view(T * N, C1, H1, W1)
            x1_upsampled = self.up(x1_reshaped)
            _, C_new, H_up, W_up = x1_upsampled.shape
            x1 = x1_upsampled.view(T, N, C_new, H_up, W_up)
        
        # input is T, N, C, H, W
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        # Pad x1 to match x2 size
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=2)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Input x is the sum of membrane potentials from the last LIFNode
        # Shape: [N, C, H, W] (temporal dimension already summed)
        return self.conv(x)

class SpikingUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, T=4):
        super(SpikingUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.T = T # Number of timesteps

        # Encoder
        self.inc = SpikingDoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256, use_attention=True) # Add attention here
        self.down3 = Down(256, 512, use_attention=True) # And here
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # The final layer is a standard Conv2d to map features to class scores.
        # It will operate on the accumulated membrane potential from the last LIF layer.
        self.outc = OutConv(64, n_classes)
        
        # We need a final LIF node whose potential we can read out
        # self.final_lif = neuron.LIFNode(surrogate_function='atan')
        self.final_lif = neuron.LIFNode()


    def forward(self, x):
        # x shape: [N, C, H, W]
        # Convert static input to a sequence of spikes over T timesteps
        # x is repeated T times along a new dimension
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # [T, N, C, H, W]

        # Reset neuron states before a new forward pass
        functional.reset_net(self)

        # Encoder path
        x1 = self.inc(x_seq)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # The output of up4 is a spike train. We pass it to the final LIF node.
        self.final_lif(x)
        
        # The output is the total membrane potential of the final LIF node,
        # which integrates the spikes over time.
        # We take the sum over the time dimension.
        potential_sum = self.final_lif.v.sum(0) # Sum over T, result shape [N, C, H, W]
        
        # Final 1x1 convolution
        logits = self.outc(potential_sum)
        
        # Apply sigmoid for binary segmentation
        return torch.sigmoid(logits)

