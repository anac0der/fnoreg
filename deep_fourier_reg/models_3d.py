import torch 
import torch.nn as nn
import torch.nn.functional as F

class SYMNet3d(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(SYMNet3d, self).__init__()
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
             
        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.r_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.r_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Dropout(0.1),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                # nn.Softsign())
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

        f_r = self.rr_dc9(r_d1)
        
        return f_r[:,0:1,:,:,:], f_r[:,1:2,:,:,:], f_r[:,2:3,:,:,:]
    
class FourierNet3d(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, patch_size):
        super(FourierNet3d, self).__init__()

        self.model = SYMNet3d(in_channel, n_classes, start_channel)
        self.patch_size = patch_size

    def field_from_outputs(self, out_1, out_2, out_3):
        if isinstance(self.patch_size, int):
            pad_size_0 = (self.patch_size - out_1.shape[-3]) // 2
            pad_size_1 = (self.patch_size - out_1.shape[-2]) // 2
            pad_size_2 = (self.patch_size - out_1.shape[-1]) // 2
        elif isinstance(self.patch_size, tuple) or isinstance(self.patch_size, list):
            pad_size_0 = (self.patch_size[0] - out_1.shape[-3]) // 2
            pad_size_1 = (self.patch_size[1] - out_1.shape[-2]) // 2
            pad_size_2 = (self.patch_size[2] - out_1.shape[-1]) // 2
        else:
            raise Exception('Wrong output size!')
        p3d = (pad_size_2, pad_size_2, pad_size_1, pad_size_1, pad_size_0, pad_size_0)

        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        out_ifft3 = torch.fft.fftshift(torch.fft.fftn(out_3))
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
        disp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
        f_xy = torch.cat([disp_mf_1, disp_mf_2, disp_mf_3], dim = 1)
        return f_xy
    
    def forward(self, x, y):
        out_1, out_2, out_3 = self.model(x, y)
        f_xy = self.field_from_outputs(out_1, out_2, out_3)
        return f_xy

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
    
class DeepUNet3d(nn.Module):
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
        self.start_layer = nn.Conv3d(in_ch, start_ch, kernel_size=3, padding='same')
        self.encoder_layers = nn.ModuleList([self.encoder_block(start_ch * 2 ** i, 
                                                                start_ch * 2 ** (i + 1)) for i in range(self.enc_blocks)])
        self.int_enc = [0 for _ in range(self.enc_blocks + 1)]
        self.latent_layer = self.latent_space_block(start_ch * 2 ** self.enc_blocks) if self.latent_space_blocks > 0 else nn.Identity()
        self.skip_conn = model_cfg['skip_conn']
        self.decoder_layers = nn.ModuleList([self.decoder_block(start_ch * 2 ** i, 
                                                                start_ch * 2 ** (i - 1),
                                                                start_ch * 2 ** (i - 1)) for i in range(self.enc_blocks, 0, -1)])

        self.end_layer = nn.Sequential(
                        nn.Conv3d(start_ch, start_ch, kernel_size=3, padding='same'),
                        nn.BatchNorm3d(start_ch),
                        nn.ReLU(),
                        nn.Conv3d(start_ch, model_cfg['out_ch'], kernel_size=1)
        )
            
    def encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            *[ConvResBlock3d(in_ch=in_ch,
                             out_ch=in_ch) for _ in range(self.enc_res_blocks)],
            
            nn.Conv3d(in_channels=in_ch, 
                      out_channels=out_ch,
                      kernel_size=3,
                      stride=2,
                      padding=1
                      ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
    
    def decoder_block(self, in_ch, out_ch, concat_size):
        return nn.ModuleDict({
            'dec_part': 
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, 
                            out_ch,
                            kernel_size=2,
                            stride=2
                            ),

                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),

                    *[ConvResBlock3d(in_ch=out_ch,
                                    out_ch=out_ch) for _ in range(self.enc_res_blocks)]),
            '1_conv': 
                nn.Sequential(
                    nn.Conv3d(out_ch + concat_size, out_ch, kernel_size=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU()) if self.skip_conn else nn.Identity()
        })
    
    def latent_space_block(self, ch):
        return nn.Sequential(
           *[ConvResBlock3d(in_ch=ch, 
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

class ConvResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding='same'):
        super().__init__()
        self.ch_reduce = nn.Conv3d(in_ch, in_ch // 2, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch // 2, out_ch // 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch // 2, out_ch // 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_ch // 2),
            nn.ReLU(),
        )
        self.ch_increase = nn.Conv3d(out_ch // 2, out_ch, kernel_size=1)
        self.rescaling = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x_in = self.ch_reduce(x)
        x_in = self.conv1(x_in)
        x_in = self.conv2(x_in)
        x_in = self.ch_increase(x_in)
        return x_in + self.rescaling(x)  