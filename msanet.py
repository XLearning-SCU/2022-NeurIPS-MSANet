import torch
from torch import nn
from mmcv.ops import ModulatedDeformConv2d

#=================================================================================================#
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, downsample=False):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, 3, 2, 1) if downsample \
            else nn.Conv2d(in_channel, out_channel, 1, 1)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        return self.conv2(self.relu(self.conv1(x))) + x

class AdaFusionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, upsample=False):
        super().__init__()
        self.conv0 = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1) if upsample \
            else nn.Conv2d(in_channel, out_channel, 1, 1)
        
        self.kernel_field = kernel_size * kernel_size
        self.offset_mask_conv = nn.Conv2d(
            out_channel*2, self.kernel_field*3, kernel_size, stride, padding
        )
        # nn.init.constant_(self.offset_mask_conv.weight, 0.)
        # nn.init.constant_(self.offset_mask_conv.bias, 0.)
        self.deform_conv = ModulatedDeformConv2d(out_channel, out_channel, kernel_size, stride, padding)
        
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, y):
        x = self.conv0(x)
        offset, mask = torch.split(
            self.offset_mask_conv(torch.cat([x, y], 1)), 
            [self.kernel_field*2, self.kernel_field], dim=1
        )
        out = x + self.deform_conv(
            y, offset.contiguous(),
            2*torch.sigmoid(mask.contiguous())
        )
        return self.conv2(self.relu(self.conv1(out))) + out

class AdaFeatBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_field = kernel_size * kernel_size
        self.offset_mask_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=self.kernel_field*3,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        # nn.init.constant_(self.offset_mask_conv.weight, 0.)
        # nn.init.constant_(self.offset_mask_conv.bias, 0.)
        self.deform_conv = ModulatedDeformConv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)

    def forward(self, x):
        offset, mask = torch.split(
            self.offset_mask_conv(x), 
            [self.kernel_field*2, self.kernel_field], dim=1
        )
        out = self.deform_conv(
            x, offset.contiguous(),
            2*torch.sigmoid(mask.contiguous())
        )
        return self.conv(self.relu(out)) + x

class AdaMScaleBlock(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv0 = nn.ModuleList([
            nn.Conv2d(in_channel, out_channel//4, 3, 1, 1, 1),
            nn.Conv2d(in_channel, out_channel//4, 3, 1, 2, 2),
            nn.Conv2d(in_channel, out_channel//4, 3, 1, 3, 3),
            nn.Conv2d(in_channel, out_channel//4, 3, 1, 4, 4)
        ])

        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_channel, out_channel), nn.Sigmoid())
        # nn.init.constant_(self.fc[0].weight, 0.)
        # nn.init.constant_(self.fc[0].bias, 0.)
        # spatial attention
        self.conv = nn.Sequential(nn.Conv2d(1, 1, 7, 1, 3), nn.Sigmoid())
        # nn.init.constant_(self.conv[0].weight, 0.)
        # nn.init.constant_(self.conv[0].bias, 0.)

        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.conv0], dim=1)
        ch_att = 2 * self.fc(self.avg_pool(out).squeeze())\
            .unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = ch_att*out
        sp_att = 2 * self.conv(torch.mean(out, dim=1, keepdim=True))
        out = sp_att*out
        return self.conv1(self.relu(out)) + x

#=================================================================================================#
class MSANet(nn.Module):
    def __init__(self, input_channel=3, output_channel=3, n_channel=32):
        super().__init__()
        
        # Encoder 
        self.conv1 = ResBlock(input_channel, n_channel, downsample=False)
        self.conv2 = ResBlock(n_channel*1, n_channel*2, downsample=True)
        self.conv3 = ResBlock(n_channel*2, n_channel*4, downsample=True)
        self.conv4 = ResBlock(n_channel*4, n_channel*8, downsample=True)
        
        # Multi-scale adaptive nets
        self.msa1 = nn.Sequential( # ABABA
            AdaFeatBlock(n_channel, n_channel),
            AdaMScaleBlock(n_channel, n_channel),
            AdaFeatBlock(n_channel, n_channel),
            AdaMScaleBlock(n_channel, n_channel),
            AdaFeatBlock(n_channel, n_channel)
        )
        self.msa2 = nn.Sequential( # ABABA
            AdaFeatBlock(n_channel*2, n_channel*2),
            AdaMScaleBlock(n_channel*2, n_channel*2),
            AdaFeatBlock(n_channel*2, n_channel*2),
            AdaMScaleBlock(n_channel*2, n_channel*2),
            AdaFeatBlock(n_channel*2, n_channel*2)
        )
        self.msa3 = nn.Sequential( # ABBA
            AdaFeatBlock(n_channel*4, n_channel*4),
            AdaMScaleBlock(n_channel*4, n_channel*4),
            AdaMScaleBlock(n_channel*4, n_channel*4),
            AdaFeatBlock(n_channel*4, n_channel*4)
        )
        self.msa4 = nn.Sequential( # BBB
            AdaMScaleBlock(n_channel*8, n_channel*8),
            AdaMScaleBlock(n_channel*8, n_channel*8),
            AdaMScaleBlock(n_channel*8, n_channel*8)
        )
        
        # Decoder
        self.dec4 = AdaFusionBlock(n_channel*8, n_channel*4, upsample=True)
        self.dec3 = AdaFusionBlock(n_channel*4, n_channel*2, upsample=True)
        self.dec2 = AdaFusionBlock(n_channel*2, n_channel*1, upsample=True)
        self.dec1 = AdaFusionBlock(n_channel, n_channel, upsample=False)
        
        self.residual = nn.Conv2d(input_channel, n_channel, 1, 1)
        self.out = nn.Conv2d(n_channel, output_channel, 1, 1)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)

        feat1 = self.msa1(feat1) + feat1
        feat2 = self.msa2(feat2) + feat2
        feat3 = self.msa3(feat3) + feat3
        feat4 = self.msa4(feat4) + feat4

        feat3 = self.dec4(feat4, feat3)
        feat2 = self.dec3(feat3, feat2)
        feat1 = self.dec2(feat2, feat1)
        feat0 = self.dec1(feat1, self.residual(x))

        return self.out(feat0)