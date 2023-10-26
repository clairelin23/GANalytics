import torch
import torch.nn as nn
import math

"""
This file defines the model architecture
"""


class ScaledLinear(nn.Module):
    """
    Scaled Fully Connected Layer
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = scale_weight(linear)

    def forward(self, x):
        return self.linear(x)


class ScaledConv2d(nn.Module):
    """
    Scaled 2D Convolution
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = scale_weight(conv)

    def forward(self, x):
        return self.conv(x)


class PixelNorm(nn.Module):
    """
    Pixel Normalization Module
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class LearnedAffineTransform(nn.Module):
    """
    Learned affine transform
    """

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = ScaledLinear(dim_latent, n_channel * 2)
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


class AdaIn(nn.Module):
    """
    Adaptive instance normalization
    """

    def __init__(self, n_channel):
        super().__init__()
        self.layer = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.layer(image)
        result = result * factor + bias
        return result


class ScaleNoise(nn.Module):
    """
    Noise Scaling
    """

    def __init__(self, n_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        result = noise * self.weight
        return result


def scale_weight(module):
    """
    Initialize the scaleWight class to apply scaling to module.
    :param module: (nn.Module) The module to which scaling is applied.
    """
    ScaleWeight.run(module)
    return module


class ScaleWeight:
    """
    Weight Scaling wrapped in a class.
    It is applied to stabilized training.
    """

    def __call__(self, module, _):
        weight = getattr(module, 'weight_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        new_weight = weight * math.sqrt(2 / fan_in)
        setattr(module, 'weight', new_weight)

    @staticmethod
    def run(module):
        """
        Apply scaling to a module.
        :param module: (nn.Module) The module to which scaling is applied.
        """
        weight = getattr(module, 'weight')
        module.register_parameter('weight_orig', nn.Parameter(weight.data))
        del module._parameters['weight']
        hook = ScaleWeight()
        module.register_forward_pre_hook(hook)


class FirstGenBlock(nn.Module):
    """
    First block of generator
    """

    def __init__(self, n_channel, dim_latent, dim_input):
        super().__init__()
        # Constant input
        self.constant = nn.Parameter(
            torch.randn(1, n_channel, dim_input, dim_input))
        self.style1 = LearnedAffineTransform(dim_latent, n_channel)
        self.style2 = LearnedAffineTransform(dim_latent, n_channel)
        self.noise1 = scale_weight(ScaleNoise(n_channel))
        self.noise2 = scale_weight(ScaleNoise(n_channel))
        self.adain = AdaIn(n_channel)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv = ScaledConv2d(n_channel, n_channel, 3, padding=1)

    def forward(self, latent_w, noise):
        result = self.constant.repeat(noise.shape[0], 1, 1, 1)
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)

        return result


class GenBlock(nn.Module):
    """
    General generator block
    """

    def __init__(self, in_channel, out_channel, dim_latent):
        super().__init__()
        self.style1 = LearnedAffineTransform(dim_latent, out_channel)
        self.style2 = LearnedAffineTransform(dim_latent, out_channel)
        self.noise1 = scale_weight(ScaleNoise(out_channel))
        self.noise2 = scale_weight(ScaleNoise(out_channel))
        self.adain = AdaIn(out_channel)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = ScaledConv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = ScaledConv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, previous_result, latent_w, noise):
        result = self.conv1(previous_result)
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv2(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)

        return result


class ConvBlock(nn.Module):
    """
    Used to construct progressive discriminator
    """

    def __init__(self, in_channel, out_channel, size_kernel1, padding1,
                 size_kernel2=None, padding2=None):
        super().__init__()

        if size_kernel2 is None:
            size_kernel2 = size_kernel1
        if padding2 is None:
            padding2 = padding1

        self.conv = nn.Sequential(
            ScaledConv2d(in_channel, out_channel, size_kernel1,
                         padding=padding1),
            nn.LeakyReLU(0.2),
            ScaledConv2d(out_channel, out_channel, size_kernel2,
                         padding=padding2),
            nn.LeakyReLU(0.2))

    def forward(self, image):
        result = self.conv(image)
        return result


# Main components
class MappingNetwork(nn.Module):
    """
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    """

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_fc):
            layers.append(ScaledLinear(dim_latent, dim_latent))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z):
        latent_w = self.mapping(latent_z)
        return latent_w


# Generator
class Generator(nn.Module):
    """
    Generator consists of mapping network and the Synthesis network
    """

    def __init__(self, n_fc, dim_latent, dim_input):
        super().__init__()
        self.fcs = MappingNetwork(n_fc, dim_latent)
        self.convolutions = nn.ModuleList([
            FirstGenBlock(512, dim_latent, dim_input),
            GenBlock(512, 512, dim_latent),
            GenBlock(512, 512, dim_latent),
            GenBlock(512, 512, dim_latent),
            GenBlock(512, 256, dim_latent),
            GenBlock(256, 128, dim_latent),
            GenBlock(128, 64, dim_latent),
            GenBlock(64, 32, dim_latent),
            GenBlock(32, 16, dim_latent)
        ])
        self.to_rgbs = nn.ModuleList([
            ScaledConv2d(512, 3, 1),
            ScaledConv2d(512, 3, 1),
            ScaledConv2d(512, 3, 1),
            ScaledConv2d(512, 3, 1),
            ScaledConv2d(256, 3, 1),
            ScaledConv2d(128, 3, 1),
            ScaledConv2d(64, 3, 1),
            ScaledConv2d(32, 3, 1),
            ScaledConv2d(16, 3, 1)
        ])

    def forward(self, latent_z, step=0, alpha=-1, noise=None,
                latent_w_center=None, psi=0):
        """
        :param latent_z: Input of mapping network
        :param step: number of layers (count from 4 x 4) used in training
        :param alpha: smooth conversion of resolution
        :param noise: Noise Input
        :param latent_w_center: Truncation trick in W
        :param psi: parameter of truncation
        """
        if type(latent_z) != type([]):
            latent_z = [latent_z]
            print('Use list to wrap latent_z, so code can be used for '
                  'style-mixing')
        latent_w = [self.fcs(latent) for latent in latent_z]

        # Truncation trick in W
        if latent_w_center:
            latent_w = [
                latent_w_center + psi * (unscaled_latent_w - latent_w_center)
                for unscaled_latent_w in latent_w]

        # Generate Gaussian noise
        result = 0
        current_latent = 0
        for i, conv in enumerate(self.convolutions):
            current_latent = latent_w[0]
            # Not the first layer, need to upsample
            if i > 0 and step > 0:
                result_upsample = nn.functional.interpolate(result,
                                                            scale_factor=2,
                                                            mode='bilinear',
                                                            align_corners=False)
                result = conv(result_upsample, current_latent, noise[i])
            else:
                result = conv(current_latent, noise[i])

            # Final layer, output rgb image
            if i == step:
                result = self.to_rgbs[i](result)
                if i > 0 and 0 <= alpha < 1:
                    result_prev = self.to_rgbs[i - 1](result_upsample)
                    result = alpha * result + (1 - alpha) * result_prev
                break
        return result


class Discriminator(nn.Module):
    """
    Discriminator model reflects similar structure complexity as Generator
    to strike an architectural balance
    """
    def __init__(self):
        super().__init__()
        self.from_rgbs = nn.ModuleList([
            ScaledConv2d(3, 16, 1),
            ScaledConv2d(3, 32, 1),
            ScaledConv2d(3, 64, 1),
            ScaledConv2d(3, 128, 1),
            ScaledConv2d(3, 256, 1),
            ScaledConv2d(3, 512, 1),
            ScaledConv2d(3, 512, 1),
            ScaledConv2d(3, 512, 1),
            ScaledConv2d(3, 512, 1)
        ])
        self.convolutions = nn.ModuleList([
            ConvBlock(16, 32, 3, 1),
            ConvBlock(32, 64, 3, 1),
            ConvBlock(64, 128, 3, 1),
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 512, 3, 1),
            ConvBlock(512, 512, 3, 1),
            ConvBlock(512, 512, 3, 1),
            ConvBlock(512, 512, 3, 1),
            ConvBlock(513, 512, 3, 1, 4, 0)
        ])
        self.fc = ScaledLinear(512, 1)
        self.n_layer = 9  # 9 layers network

    def forward(self, image, step=0, alpha=-1):
        """
        :param image: generator output
        :param step: number of layers (count from 4 x 4) used in training
        :param alpha: smooth conversion of resolution
        """
        for i in range(step, -1, -1):
            # Gets the index of current layer, count from the layer 4 * 4
            layer_index = self.n_layer - i - 1
            if i == step: # First layer, we use from_rgb to convert to n_channel data
                result = self.from_rgbs[layer_index](image)
            # Do minibatch stddev before final layer
            if i == 0:
                res_var = result.var(0, unbiased=False) + 1e-8  # To avoid zero
                res_std = torch.sqrt(res_var)
                mean_std = res_std.mean().expand(result.size(0), 1, 4, 4)
                result = torch.cat([result, mean_std], 1)
            result = self.convolutions[layer_index](result)

            # Not final layer
            if i > 0:
                # Down-sample for further usage
                result = nn.functional.interpolate(result, scale_factor=0.5,
                                                   mode='bilinear',
                                                   align_corners=False)
                # Alpha set, combine the result of different layers when input
                if i == step and 0 <= alpha < 1:
                    result_next = self.from_rgbs[layer_index + 1](image)
                    result_next = nn.functional.interpolate(result_next,
                                                            scale_factor=0.5,
                                                            mode='bilinear',
                                                            align_corners=False)
                    result = alpha * result + (1 - alpha) * result_next

        result = result.squeeze(2).squeeze(2)
        result = self.fc(result)
        return result
