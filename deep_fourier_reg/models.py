import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ffc_blocks import FFCResNetBlock

# Fourier-Net implementation from https://github.com/xi-jia/Fourier-Net

class SYMNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True
        self.bn = False
        super(SYMNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.rr_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.r_dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
             
        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8, batchnorm=False)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4, batchnorm=False)
        self.r_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2, batchnorm=False)
        self.r_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2, batchnorm=False)
        

        # self.i_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        # self.i_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.i_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.i_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.ii_dc9 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.i_dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
             
        # self.i_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        # self.i_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        # self.i_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # self.i_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, output_padding=output_padding, bias=bias),
                nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                #nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)

        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)

        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        # r_d2 = torch.cat((self.r_up3(r_d1), e1), 1)

        # r_d2 = self.r_dc5(r_d2)
        # r_d2 = self.r_dc6(r_d2)

        # r_d2 = torch.cat((self.r_up3(r_d1), e1), 1)

        # r_d2 = self.r_dc5(r_d2)
        # r_d2 = self.r_dc6(r_d2)

        # r_d3 = torch.cat((self.r_up4(r_d2), e0), 1)
        # r_d3 = self.r_dc7(r_d3)
        # r_d3 = self.r_dc8(r_d3)

        # f_r = self.rr_dc9(r_d3)
        # print('r_d2.shape   ', r_d2.shape)
        f_r = self.rr_dc9(r_d1) * 64
        

        # i_d0 = torch.cat((self.i_up1(e4), e3), 1)

        # i_d0 = self.i_dc1(i_d0)
        # i_d0 = self.i_dc2(i_d0)

        # i_d1 = torch.cat((self.i_up2(i_d0), e2), 1)

        # i_d1 = self.i_dc3(i_d1)
        # i_d1 = self.i_dc4(i_d1)

        # i_d2 = torch.cat((self.i_up3(i_d1), e1), 1)

        # i_d2 = self.i_dc5(i_d2)
        # i_d2 = self.i_dc6(i_d2)

        # i_d3 = torch.cat((self.i_up4(i_d2), e0), 1)
        # i_d3 = self.i_dc7(i_d3)
        # i_d3 = self.i_dc8(i_d3)

        # f_i = self.ii_dc9(i_d3)
        # f_i = self.ii_dc9(i_d0) * 64
        
        
        return f_r[:,0:1,:,:], f_r[:,1:2,:,:]
        # return torch.complex(f_r[:,0:1,:,:], f_i[:,0:1,:,:]), torch.complex(f_r[:,1:2,:,:], f_i[:,1:2,:,:])

        # return f_xy#, f_yx

class SYMNetFull(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, ffc=0):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True
        self.bn = False
        super(SYMNetFull, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.rr_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.r_dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
             
        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8, batchnorm=False)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4, batchnorm=False)
        self.r_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2, batchnorm=False)
        self.r_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2, batchnorm=False)

        self.latent_space_block = self.ffc_block(self.start_channel * 16, 0.5, ffc) if ffc > 0 else nn.Identity()
        

        # self.i_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        # self.i_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.i_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.i_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  # stride=1, bias=bias_opt)
        # self.i_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.ii_dc9 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.i_dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
             
        # self.i_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        # self.i_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        # self.i_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # self.i_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, output_padding=output_padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, output_padding=output_padding, bias=bias),
                nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                #nn.Softsign())
        return layer

    def ffc_block(self, ch, alpha, num):
        return nn.Sequential(
           *[FFCResNetBlock(in_channels=ch, 
                            out_channels=ch,
                            alpha_in=alpha,
                             alpha_out=alpha,
                             kernel_size=3,
                             fu_kernel=3, 
                             use_only_freq=False)
                            for _ in range(num)])


    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)

        e4 = self.latent_space_block(e4)
        
        e4 = self.ec9(e4)

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)

        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)

        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        r_d2 = torch.cat((self.r_up3(r_d1), e1), 1)

        r_d2 = self.r_dc5(r_d2)
        r_d2 = self.r_dc6(r_d2)

        # r_d2 = torch.cat((self.r_up3(r_d1), e1), 1)

        # r_d2 = self.r_dc5(r_d2)
        # r_d2 = self.r_dc6(r_d2)

        r_d3 = torch.cat((self.r_up4(r_d2), e0), 1)
        r_d3 = self.r_dc7(r_d3)
        r_d3 = self.r_dc8(r_d3)

        f_r = self.rr_dc9(r_d3)
        # print('r_d2.shape   ', r_d2.shape)
        # f_r = self.rr_dc9(r_d1) * 64
        

        # i_d0 = torch.cat((self.i_up1(e4), e3), 1)

        # i_d0 = self.i_dc1(i_d0)
        # i_d0 = self.i_dc2(i_d0)

        # i_d1 = torch.cat((self.i_up2(i_d0), e2), 1)

        # i_d1 = self.i_dc3(i_d1)
        # i_d1 = self.i_dc4(i_d1)

        # i_d2 = torch.cat((self.i_up3(i_d1), e1), 1)

        # i_d2 = self.i_dc5(i_d2)
        # i_d2 = self.i_dc6(i_d2)

        # i_d3 = torch.cat((self.i_up4(i_d2), e0), 1)
        # i_d3 = self.i_dc7(i_d3)
        # i_d3 = self.i_dc8(i_d3)

        # f_i = self.ii_dc9(i_d3)
        # f_i = self.ii_dc9(i_d0) * 64
        
        
        return torch.cat([f_r[:,0:1,:,:], f_r[:,1:2,:,:]], 1)
        # return torch.complex(f_r[:,0:1,:,:], f_i[:,0:1,:,:]), torch.complex(f_r[:,1:2,:,:], f_i[:,1:2,:,:])

        # return f_xy#, f_yx
class VxmModel(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, ffc=0):
        super(VxmModel, self).__init__()
        self.model = SYMNetFull(in_channel, n_classes, start_channel, ffc)
        self.spatial_transform = SpatialTransform()
        for param in self.spatial_transform.parameters():
            param.requires_grad = False
            param.volatile = True
    
    def forward(self, x, y):
        out1, out2 = self.model(x, y)
        f_xy = torch.cat([out1, out2], 1)
        grid, X_Y = self.spatial_transform(x, f_xy.permute(0, 2, 3, 1))
        return grid, f_xy, X_Y

class FourierNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, patch_size):
        super(FourierNet, self).__init__()
        self.model = SYMNet(in_channel, n_classes, start_channel)
        self.spatial_transform = SpatialTransform()
        for param in self.spatial_transform.parameters():
            param.requires_grad = False
            param.volatile = True
        self.patch_size = patch_size
    
    def field_from_outputs(self, out_1, out_2):
        out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))
        if isinstance(self.patch_size, int):
            pad_size_0 = (self.patch_size - out_1.shape[-2]) // 2
            pad_size_1 = (self.patch_size - out_1.shape[-1]) // 2
        elif isinstance(self.patch_size, tuple) or isinstance(self.patch_size, list):
            pad_size_0 = (self.patch_size[0] - out_1.shape[-2]) // 2
            pad_size_1 = (self.patch_size[1] - out_1.shape[-1]) // 2
        else:
            raise Exception('Wrong output size!')
        p3d = (pad_size_1, pad_size_1, pad_size_0, pad_size_0)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)

        disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))
        # print(disp_mf_1.shape, disp_mf_2.shape)
        f_xy = torch.cat([disp_mf_1, disp_mf_2], dim = 1)
        return f_xy
    
    def forward(self, x, y):
        out_1, out_2 = self.model(x, y)
        f_xy = self.field_from_outputs(out_1, out_2)
        # grid, X_Y = self.spatial_transform(x, f_xy.permute(0, 2, 3, 1))
        return f_xy      

class FourierNetDiff(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, input_size):
        super(FourierNetDiff, self).__init__()
        self.model = SYMNet(in_channel, n_classes, start_channel)
        self.spatial_transform = SpatialTransform()
        self.diff_transform = DiffeomorphicTransform()
        for param in self.spatial_transform.parameters():
            param.requires_grad = False
            param.volatile = True
        self.input_size = input_size
    
    def field_from_outputs(self, out_1, out_2):
        out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))

        pad_size_0 = (self.input_size - out_1.shape[-2]) // 2
        pad_size_1 = (self.input_size - out_1.shape[-1]) // 2
        p3d = (pad_size_0, pad_size_0, pad_size_1, pad_size_1)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)

        disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))
        # print(disp_mf_1.shape, disp_mf_2.shape)
        f_xy = torch.cat([disp_mf_1, disp_mf_2], dim = 1)
        return f_xy
    
    def forward(self, x, y):
        out_1, out_2 = self.model(x, y)
        f_xy = self.field_from_outputs(out_1, out_2)
        D_f_xy = self.diff_transform(f_xy)
        _, X_Y = self.spatial_transform(x, D_f_xy.permute(0, 2, 3, 1))
        return D_f_xy, X_Y 

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
    
        # print(flow.shape)
        h2, w2 = flow.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        
        
        for i in range(self.time_step):
            flow_h = flow[:,0,:,:]
            flow_w = flow[:,1,:,:]
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w,disp_h), dim=3)
    
            # print(deformation.shape)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear',padding_mode="border", align_corners = True)
            #Softsign
            #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
            #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
            #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
            
            # Remove Channel Dimension
            #disp_h = (grid_h + (flow_h)).squeeze(1)
            #disp_w = (grid_w + (flow_w)).squeeze(1)

            #sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, H, W, 2)
            #flow = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod)
        
        return flow

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        # d2, h2, w2 = mov_image.shape[-3:]
        h2, w2 = mov_image.shape[-2:]
        # grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        # grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        # grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        # flow_w = flow[:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        # disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return sample_grid, warped

class SpatialTransform3d(nn.Module):
    def __init__(self):
        super(SpatialTransform3d, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:,:,:,:,0]
        flow_h = flow[:,:,:,:,1]
        flow_w = flow[:,:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return sample_grid, warped

class DeepUNet2d(nn.Module):
    def __init__(
            self,
            model_cfg   
    ):
        super().__init__()
        self.start_ch = model_cfg['start_ch']
        self.enc_blocks = model_cfg['enc_blocks']
        self.latent_space_blocks = model_cfg['latent_space_blocks']
        self.enc_res_blocks = model_cfg['enc_res_blocks']

        in_ch = model_cfg['in_ch']
        start_ch = model_cfg['start_ch']
        self.start_layer = nn.Conv2d(in_ch, start_ch, kernel_size=3, padding='same')
        self.encoder_layers = nn.ModuleList([self.encoder_block(start_ch * 2 ** i, 
                                                                start_ch * 2 ** (i + 1)) for i in range(self.enc_blocks)])
        self.int_enc = [0 for _ in range(self.enc_blocks + 1)]
        self.latent_layer = self.latent_space_block(start_ch * 2 ** self.enc_blocks) if self.latent_space_blocks > 0 else nn.Identity()
        self.skip_conn = model_cfg['skip_conn']
        self.decoder_layers = nn.ModuleList([self.decoder_block(start_ch * 2 ** i, 
                                                                start_ch * 2 ** (i - 1),
                                                                start_ch * 2 ** (i - 1)) for i in range(self.enc_blocks, 0, -1)])

        self.end_layer = nn.Sequential(
                        nn.Conv2d(start_ch, start_ch, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(start_ch),
                        nn.ReLU(),
                        nn.Conv2d(start_ch, model_cfg['out_ch'], kernel_size=1)
        )
            
    def encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            *[ConvResBlock2d(in_ch=in_ch,
                             out_ch=in_ch) for _ in range(self.enc_res_blocks)],
            
            nn.Conv2d(in_channels=in_ch, 
                      out_channels=out_ch,
                      kernel_size=3,
                      stride=2,
                      padding=1
                      ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def decoder_block(self, in_ch, out_ch, concat_size):
        return nn.ModuleDict({
            'dec_part': 
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, 
                            out_ch,
                            kernel_size=2,
                            stride=2
                            ),

                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),

                    *[ConvResBlock2d(in_ch=out_ch,
                                    out_ch=out_ch) for _ in range(self.enc_res_blocks)]),
            '1_conv': 
                nn.Sequential(
                    nn.Conv2d(out_ch + concat_size, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU()) if self.skip_conn else nn.Identity()
        })
    
    def latent_space_block(self, ch):
        return nn.Sequential(
           *[ConvResBlock2d(in_ch=ch, 
                            out_ch=ch)
                            for _ in range(self.latent_space_blocks)])

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        x_in = self.start_layer(x_in)
        self.int_enc[0] = x_in
        for i, layer in enumerate(self.encoder_layers):
            x_in = layer(x_in)
            self.int_enc[i + 1] = x_in
        
        x_in = self.latent_layer(x_in)
        for i, layer in enumerate(self.decoder_layers):
            x_in = layer['dec_part'](x_in)
            if self.skip_conn:
                x_in = torch.cat((x_in, self.int_enc[- i - 2]), 1)
                x_in = layer['1_conv'](x_in)
        
        return self.end_layer(x_in)

class ConvResBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding='same'):
        super().__init__()
        self.ch_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch // 2, out_ch // 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(),
        )
        self.ch_increase = nn.Conv2d(out_ch // 2, out_ch, kernel_size=1)
        self.rescaling = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x_in = self.ch_reduce(x)
        x_in = self.conv1(x_in)
        x_in = self.conv2(x_in)
        x_in = self.ch_increase(x_in)
        return x_in + self.rescaling(x)
    
def smoothloss(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    h2, w2 = y_pred.shape[-2:]
    # dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0, dx, dz

def smoothloss3d(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    d2, h2, w2 = y_pred.shape[-3:]
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0, dx, dy, dz

def boundary_loss3d(dx, dy, dz):
    top_boundary = dy[:, :, 0, :, :]
    bottom_boundary = dy[:, :, -1, :, :]
    back_boundary = dx[:, :, :, 0, :]
    front_boundary = dx[:, :, :, -1, :]
    left_boundary = dz[:, :, :, :, 0]
    right_boundary = dz[:, :, :, :, -1]
    boundary = torch.cat([torch.flatten(top_boundary), torch.flatten(bottom_boundary), torch.flatten(left_boundary), 
                            torch.flatten(right_boundary), torch.flatten(back_boundary), torch.flatten(front_boundary)], dim=-1)
    boundary_sq  = boundary * boundary
    boundary_loss = torch.mean(torch.sum(boundary_sq, dim=-1))
    return boundary_loss

def gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)
    
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    
    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = (right - left) * w_x, (bottom - top) * h_x
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    
    return dx, dy

def deformation_smooth_loss(flow):
    """
        Computes a deformation smoothness based loss as described here:
        https://link.springer.com/content/pdf/10.1007%2F978-3-642-33418-4_16.pdf
        """
    
    dx, dy = gradient(flow)
    
    dx2, dxy = gradient(dx)
    dyx, dy2 = gradient(dy)
    
    integral = torch.mul(dx2, dx2) + torch.mul(dy2, dy2) + torch.mul(dxy, dxy) + torch.mul(dyx, dyx)
#    loss = torch.sum(integral, [1,2,3]).mean()
    loss = torch.mean(integral)
    return loss, dx, dy

"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class MSE(torch.nn.Module):
    """
    Mean squared error loss.
    """

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
    
class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class ModelConfig:
    def __init__(self, model_params):
        self.model_params = model_params
    
    def __getitem__(self, key):
        try:
            return self.model_params[key]
        except Exception:
            raise Exception(f'Key {key} does not exist in current config!')
    
    def get_params(self):
        return self.model_params
    