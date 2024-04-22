import torch
import torch.nn as nn
from ffc_blocks import FFCResNetBlock
from neuralop.models import FNO

class FFCUnet(nn.Module):
    def __init__(
            self,
            model_cfg   
    ):
        super().__init__()
        self.start_ch = model_cfg['start_ch']
        self.fu_kernel = model_cfg['fu_kernel']
        self.kernel_size = model_cfg['kernel_size']
        self.enc_blocks = model_cfg['enc_blocks']
        self.latent_space_blocks = model_cfg['latent_space_blocks']
        self.enc_res_blocks = model_cfg['enc_res_blocks']

        start_alpha = model_cfg['start_alpha']
        end_alpha = model_cfg['end_alpha']
        h_alpha = (start_alpha - end_alpha) // self.enc_blocks
        in_ch = model_cfg['in_ch']
        start_ch = model_cfg['start_ch']
        self.start_layer = nn.Conv2d(in_ch, start_ch, kernel_size=3, padding='same')
        self.encoder_layers = nn.ModuleList([self.encoder_block(start_ch * 2 ** i, 
                                                                start_ch * 2 ** (i + 1), 
                                                                start_alpha - i * h_alpha) for i in range(self.enc_blocks)])
        self.int_enc = [0 for _ in range(self.enc_blocks + 1)]
        self.latent_layer = self.latent_space_block(start_ch * 2 ** self.enc_blocks, end_alpha) if self.latent_space_blocks > 0 else nn.Identity()
        self.skip_conn = model_cfg['skip_conn']
        self.decoder_layers = nn.ModuleList([self.decoder_block(start_ch * 2 ** i, 
                                                                start_ch * 2 ** (i - 1),
                                                                start_ch * 2 ** (i - 1),
                                                                start_alpha - i * h_alpha) for i in range(self.enc_blocks, 0, -1)])

        self.end_layer = nn.Sequential(
                        nn.Conv2d(start_ch, start_ch, kernel_size=3, padding='same'),
                        nn.BatchNorm2d(start_ch),
                        nn.ReLU(),
                        nn.Conv2d(start_ch, model_cfg['out_ch'], kernel_size=1)
        )
            
    def encoder_block(self, in_ch, out_ch, alpha):
        return nn.Sequential(
            *[FFCResNetBlock(in_channels=in_ch,
                             out_channels=in_ch,
                             alpha_in=alpha,
                             alpha_out=alpha,
                             kernel_size=self.kernel_size,
                             fu_kernel=self.fu_kernel, 
                             use_only_freq=False) for _ in range(self.enc_res_blocks)],
            
            nn.Conv2d(in_channels=in_ch, 
                      out_channels=out_ch,
                      kernel_size=3,
                      stride=2,
                      padding=1
                      ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def decoder_block(self, in_ch, out_ch, concat_size, alpha):
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

                    *[FFCResNetBlock(in_channels=out_ch,
                                    out_channels=out_ch,
                                    alpha_in=alpha,
                                    alpha_out=alpha,
                                    kernel_size=self.kernel_size,
                                    fu_kernel=self.fu_kernel, 
                                    use_only_freq=False) for _ in range(self.enc_res_blocks)]),
            '1_conv': 
                nn.Sequential(
                    nn.Conv2d(out_ch + concat_size, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU()) if self.skip_conn else nn.Identity()
        })
    
    def latent_space_block(self, ch, alpha):
        return nn.Sequential(
           *[FFCResNetBlock(in_channels=ch, 
                            out_channels=ch,
                            alpha_in=alpha,
                             alpha_out=alpha,
                             kernel_size=self.kernel_size,
                             fu_kernel=self.fu_kernel, 
                             use_only_freq=False)
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

class ConvResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding='same'):
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

class FFCAE(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            start_ch: int,
            alpha: float,
            enc_blocks,
            latent_space_blocks,
            kernel_size = 3,
            fu_kernel = 1,      
    ):  
        super().__init__()
        self.enc_blocks = enc_blocks
        self.latent_space_blocks = latent_space_blocks
        self.kernel_size = kernel_size
        self.fu_kernel = fu_kernel

        self.start_layer = nn.Sequential(
            nn.Conv2d(in_ch, start_ch, kernel_size=1),
            nn.ReLU(),
        )
        self.enc = nn.Sequential(*[self.encoder_block(start_ch * 2 ** i, 
                                                      start_ch * 2 ** (i + 1)) for i in range(self.enc_blocks)])
        self.ffc_layers = self.latent_space_layer(start_ch * 2 ** self.enc_blocks, alpha)
        self.dec = nn.Sequential(*[self.decoder_block(start_ch * 2 ** i, 
                                                      start_ch * 2 ** (i - 1)) for i in range(self.enc_blocks, 0, -1)])
        self.end_layer = nn.Sequential(
            nn.Conv2d(start_ch, start_ch, kernel_size=3, padding='same'),
            nn.BatchNorm2d(start_ch),
            nn.ReLU(),
            nn.Conv2d(start_ch, out_ch, kernel_size=1, padding='same')
        )
    
    def encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            ConvResBlock(in_ch, in_ch, kernel_size=self.kernel_size, padding='same'),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
        )
    def decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ConvResBlock(out_ch, out_ch, kernel_size=self.kernel_size, padding='same'),
        )

    def latent_space_layer(self, ch, alpha):
        return nn.Sequential(
           *[FFCResNetBlock(in_channels=ch, 
                            out_channels=ch,
                            alpha_in=alpha,
                             alpha_out=alpha,
                             kernel_size=self.kernel_size,
                             fu_kernel=self.fu_kernel, 
                             use_only_freq=False)
                            for _ in range(self.latent_space_blocks)])

    def forward(self, x, y):
        x_in = torch.cat([x, y], 1)
        x_in = self.start_layer(x_in)
        x_in = self.enc(x_in)
        x_in = self.ffc_layers(x_in)
        x_in = self.dec(x_in)
        x_in = self.end_layer(x_in)

        return x_in            