import torch
import torch.nn as nn

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class FourierUnit(torch.nn.Module):
    """Implements Fourier Unit block.

    Applies FFT to tensor and performs convolution in spectral domain.
    After that return to time domain with Inverse FFT.

    Attributes:
        inter_conv: conv-bn-relu block that performs conv in spectral domain

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fu_kernel: int = 1,
        padding_type: str = "reflect",
        fft_norm: str = "ortho",
        use_only_freq: bool = False,
        norm_layer=nn.BatchNorm2d,
        bias: bool = True,
    ):
        super().__init__()
        self.fft_norm = fft_norm
        self.use_only_freq = use_only_freq

        self.inter_conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                out_channels * 2,
                kernel_size=fu_kernel,
                stride=1,
                padding=get_padding(fu_kernel),
                padding_mode=padding_type,
                bias=bias,
            ),
            norm_layer(out_channels * 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        batch_size, ch, freq_dim, embed_dim = x.size()

        dims_to_fft = (-2,) if self.use_only_freq else (-2, -1)
        recover_length = (freq_dim,) if self.use_only_freq else (freq_dim, embed_dim)

        fft_representation = torch.fft.rfftn(x, dim=dims_to_fft, norm=self.fft_norm)

        # (B, Ch, 2, FFT_freq, FFT_embed)
        fft_representation = torch.stack(
            (fft_representation.real, fft_representation.imag), dim=2
        )  # .view(batch_size, ch * 2, -1, embed_dim)

        ffted_dims = fft_representation.size()[-2:]
        fft_representation = fft_representation.view(
            (
                batch_size,
                ch * 2,
            )
            + ffted_dims
        )

        fft_representation = (
            self.inter_conv(fft_representation)
            .view(
                (
                    batch_size,
                    ch,
                    2,
                )
                + ffted_dims
            )
            .permute(0, 1, 3, 4, 2)
        )

        fft_representation = torch.complex(
            fft_representation[..., 0], fft_representation[..., 1]
        )

        reconstructed_x = torch.fft.irfftn(
            fft_representation, dim=dims_to_fft, s=recover_length, norm=self.fft_norm
        )

        # assert reconstructed_x.size() == x.size()

        return reconstructed_x


class SpectralTransform(torch.nn.Module):
    """Implements Spectrals Transform block.

    Residual Block containing Fourier Unit with convolutions before and after.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fu_kernel: int = 1,
        padding_type: str = "reflect",
        fft_norm: str = "ortho",
        use_only_freq: bool = False,
        norm_layer=nn.BatchNorm2d,
        bias: bool = False,
    ):
        super().__init__()
        halved_out_ch = out_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, halved_out_ch, kernel_size=1, stride=1, bias=bias),
            norm_layer(halved_out_ch),
            nn.ReLU(True),
        )

        self.fu = FourierUnit(
            halved_out_ch,
            halved_out_ch,
            fu_kernel=fu_kernel,
            use_only_freq=use_only_freq,
            fft_norm=fft_norm,
            padding_type=padding_type,
            norm_layer=norm_layer,
        )

        self.conv2 = nn.Conv2d(
            halved_out_ch, out_channels, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x):

        residual = self.conv1(x)
        x = self.fu(residual)
        x += residual
        x = self.conv2(x)

        return x


class FastFourierConvolution(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha_in: float = 0.5,
        alpha_out: float = 0.5,
        kernel_size: int = 3,
        padding_type: str = "reflect",
        fu_kernel: int = 1,
        fft_norm: str = "ortho",
        bias: bool = True,
        norm_layer=nn.BatchNorm2d,
        activation=nn.ReLU(True),
        use_only_freq: bool = False,
    ):
        
        super().__init__()
        self.global_in_channels = int(in_channels * alpha_in)
        self.local_in_channels = in_channels - self.global_in_channels
        self.global_out_channels = int(out_channels * alpha_out)
        self.local_out_channels = out_channels - self.global_out_channels

        padding = get_padding(kernel_size)

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.local_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.l2l_layer = tmp_module(
            self.local_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.global_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.l2g_layer = tmp_module(
            self.local_in_channels,
            self.global_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.local_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.g2l_layer = tmp_module(
            self.global_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.global_out_channels > 0,
            SpectralTransform,
            nn.Identity,
        )
        self.g2g_layer = tmp_module(
            self.global_in_channels,
            self.global_out_channels,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            padding_type=padding_type,
            bias=bias,
            norm_layer=norm_layer,
            use_only_freq=use_only_freq,
        )

        self.local_bn_relu = (
            nn.Sequential(norm_layer(self.local_out_channels), activation)
            if self.local_out_channels != 0
            else nn.Identity()
        )

        self.global_bn_relu = (
            nn.Sequential(norm_layer(self.global_out_channels), activation)
            if self.global_out_channels != 0
            else nn.Identity()
        )

    @staticmethod
    def _get_module_on_true_predicate(
        condition: bool, true_module=nn.Identity, false_module=nn.Identity
    ):
        if condition:
            return true_module
        else:
            return false_module

    def forward(self, x):

        #  chunk into local and global channels
        x_l, x_g = (
            x[:, : self.local_in_channels, ...],
            x[:, self.local_in_channels :, ...],
        )
        x_l = 0 if x_l.size()[1] == 0 else x_l
        x_g = 0 if x_g.size()[1] == 0 else x_g

        out_local, out_global = torch.Tensor(0).to(x.device), torch.Tensor(0).to(
            x.device
        )

        if self.local_out_channels != 0:
            out_local = self.l2l_layer(x_l) + self.g2l_layer(x_g)
            out_local = self.local_bn_relu(out_local)

        if self.global_out_channels != 0:
            out_global = self.l2g_layer(x_l) + self.g2g_layer(x_g)
            out_global = self.global_bn_relu(out_global)

        #  (B, out_ch, F, T)
        output = torch.cat((out_local, out_global), dim=1)

        return output


class FFCResNetBlock(torch.nn.Module):
    """Implements Residual FFC block.

    Contains two FFC blocks with residual connection.

    Wraps around FFC arguments.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha_in: float = 0.5,
        alpha_out: float = 0.5,
        kernel_size: int = 3,
        padding_type: str = "reflect",
        bias: bool = True,
        fu_kernel: int = 1,
        fft_norm: str = "ortho",
        use_only_freq: bool = True,
        norm_layer=nn.BatchNorm2d,
        activation=nn.ReLU(True),
    ):
        super().__init__()
        self.ffc1 = FastFourierConvolution(
            in_channels,
            out_channels,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            kernel_size=kernel_size,
            padding_type=padding_type,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            use_only_freq=use_only_freq,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
        )

        self.ffc2 = FastFourierConvolution(
            in_channels,
            out_channels,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            kernel_size=kernel_size,
            padding_type=padding_type,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            use_only_freq=use_only_freq,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
        )

    def forward(self, x):
        out = self.ffc1(x)
        out = self.ffc2(out)
        return x + out
    
class FourierUnit3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fu_kernel: int = 1,
        padding_type: str = "reflect",
        fft_norm: str = "ortho",
        norm_layer=nn.BatchNorm3d,
        bias: bool = True,
    ):
        super().__init__()
        self.fft_norm = fft_norm

        self.inter_conv = nn.Sequential(
            nn.Conv3d(
                in_channels * 2,
                out_channels * 2,
                kernel_size=fu_kernel,
                stride=1,
                padding=get_padding(fu_kernel),
                padding_mode=padding_type,
                bias=bias,
            ),
            norm_layer(out_channels * 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        batch_size, ch, dim1, dim2, dim3 = x.size()

        dims_to_fft = (-3, -2, -1)
        recover_length = (dim1, dim2, dim3)

        fft_representation = torch.fft.rfftn(x, dim=dims_to_fft, norm=self.fft_norm)

        # (B, Ch, 2, FFT_freq, FFT_embed)
        fft_representation = torch.stack(
            (fft_representation.real, fft_representation.imag), dim=2
        )  # .view(batch_size, ch * 2, -1, embed_dim)

        ffted_dims = fft_representation.size()[-3:]
        fft_representation = fft_representation.view(
            (
                batch_size,
                ch * 2,
            )
            + ffted_dims
        )

        fft_representation = (
            self.inter_conv(fft_representation)
            .view(
                (
                    batch_size,
                    ch,
                    2,
                )
                + ffted_dims
            )
            .permute(0, 1, 3, 4, 5, 2)
        )

        fft_representation = torch.complex(
            fft_representation[..., 0], fft_representation[..., 1]
        )

        reconstructed_x = torch.fft.irfftn(
            fft_representation, dim=dims_to_fft, s=recover_length, norm=self.fft_norm
        )

        # assert reconstructed_x.size() == x.size()

        return reconstructed_x

class SpectralTransform3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fu_kernel: int = 1,
        padding_type: str = "reflect",
        fft_norm: str = "ortho",
        norm_layer=nn.BatchNorm3d,
        bias: bool = False,
        activation=nn.ReLU(True)
    ):
        super().__init__()
        halved_out_ch = out_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, halved_out_ch, kernel_size=1, stride=1, bias=bias),
            norm_layer(halved_out_ch),
            activation
        )

        self.fu = FourierUnit3d(
            halved_out_ch,
            halved_out_ch,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            padding_type=padding_type,
            norm_layer=norm_layer,
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(halved_out_ch, out_channels, kernel_size=1, stride=1, bias=bias),
            norm_layer(out_channels),
            activation
        )

    def forward(self, x):

        residual = self.conv1(x)
        x = self.fu(residual)
        x += residual
        x = self.conv2(x)

        return x
    
class STResNetBlock3d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding_type: str = "reflect",
        bias: bool = True,
        fu_kernel: int = 1,
        fft_norm: str = "ortho",
        norm_layer=nn.BatchNorm3d,
        activation=nn.ReLU(True),
    ):
        super().__init__()
        self.st1 = SpectralTransform3d(
            in_channels,
            out_channels,
            padding_type=padding_type,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
        )

        self.st2 = SpectralTransform3d(
            in_channels,
            out_channels,
            padding_type=padding_type,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
        )

    def forward(self, x):
        out = self.st1(x)
        out = self.st2(out)
        return x + out