import torch
from torch import nn
from neuralop.models import FNO
from neuralop.models.spectral_convolution import FactorizedSpectralConv2d, FactorizedSpectralConv3d
from fourier_blocks import FFCResNetBlock, STResNetBlock3d

class MyFNO(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model = FNO(**model_cfg)
    
    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        return self.model(x_in)

class FourierLayer2d(nn.Module):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5, nonlinearity=nn.GELU):
        super().__init__()
        
        self.spectral_conv = FactorizedSpectralConv2d(in_ch, 
                                            out_ch, 
                                            n_modes, 
                                            factorization=factorization,
                                            rank=rank)
        
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nonlinearity()
    
    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_skip = self.skip(x).to(x.device)
        x_act = self.act(x_skip + x_fc)
        return x_act + x
    
class FNOReg(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv2d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)
        
        alpha = 1

        self.encoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    kernel_size=3,
                                    fu_kernel=1,
                                    alpha_in=alpha,
                                    alpha_out=alpha                                      
                                    )
        
        fact = model_cfg['factorization']
        fact = None if not fact else fact
        self.fno_blocks = nn.Sequential(
            *[
                FourierLayer2d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank']
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )

        self.decoder = FFCResNetBlock(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    kernel_size=3,
                                    fu_kernel=1,
                                    alpha_in=alpha,
                                    alpha_out=alpha                                      
                                    )
        
        self.projection = nn.Sequential(
            nn.Conv2d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )
    
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)

class FourierLayer3d(nn.Module):
    def __init__(self, in_ch, out_ch, n_modes, factorization=None, rank=0.5, nonlinearity=nn.GELU):
        super().__init__()
        
        self.spectral_conv = FactorizedSpectralConv3d(in_ch, 
                                            out_ch, 
                                            n_modes, 
                                            factorization=factorization,
                                            rank=rank)
        
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.act = nonlinearity()
    
    def forward(self, x):
        x_fc = self.spectral_conv(x)
        x_skip = self.skip(x).to(x.device)
        x_act = self.act(x_skip + x_fc)
        return x_act + x
    
class FNOReg3d(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.lifting = nn.Conv3d(model_cfg['in_channels'], model_cfg['hidden_channels'], kernel_size=1)
    
        self.encoder = STResNetBlock3d(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    fu_kernel=1                                     
                                    )
        
        fact = model_cfg['factorization']
        fact = None if not fact else fact
        self.fno_blocks = nn.Sequential(
            *[
                FourierLayer3d(
                    in_ch=model_cfg['hidden_channels'],
                    out_ch=model_cfg['hidden_channels'],
                    n_modes=model_cfg['n_modes'],
                    factorization=fact,
                    rank=model_cfg['rank']
                )
                for _ in range(model_cfg['n_layers'])
            ]
        )
        
        self.decoder = STResNetBlock3d(in_channels=model_cfg['hidden_channels'],
                                    out_channels=model_cfg['hidden_channels'],
                                    fu_kernel=1                                   
                                    )
        
        self.projection = nn.Sequential(
            nn.Conv3d(model_cfg['hidden_channels'],
                      model_cfg['projection_channels'],
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(model_cfg['projection_channels'],
                      model_cfg['out_channels'],
                      kernel_size=1)
        )
    
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.lifting(x)
        x = self.encoder(x)
        x = self.fno_blocks(x)
        x = self.decoder(x)
        return self.projection(x)