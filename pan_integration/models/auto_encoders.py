import math
import torch
from torch import nn
from typing import List

# cheeky way to make calculations more readable
i = {"channels": 0, "kernel": 1, "stride": 2, "padding": 3}


class _ConvEncoder(nn.Module):
    def __init__(self, conv_layer_sizes: List, activation_fn):
        super().__init__()
        self.blocks = nn.ModuleList()
        for layer_in, layer_out in zip(conv_layer_sizes, conv_layer_sizes[1::]):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.Conv2d(
                            in_channels=layer_in[i["channels"]],
                            out_channels=layer_out[i["channels"]],
                            kernel_size=layer_out[i["kernel"]],
                            stride=layer_out[i["stride"]],
                            padding=layer_out[i["padding"]],
                        ),
                        "bn": nn.BatchNorm2d(layer_out[i["channels"]]),
                        "activation": activation_fn,
                    }
                )
            )

        self.blocks[-1]["activation"] = nn.Sigmoid()

    def forward(self, x):
        for block in self.blocks:
            x = block["conv"](x)
            x = block["bn"](x)
            x = block["activation"](x)

        return x


class _ConvDecoder(nn.Module):
    def __init__(self, conv_layer_sizes, activation_fn, conv_out_sizes):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.conv_out_sizes = conv_out_sizes

        for layer_in, layer_out in zip(
            conv_layer_sizes[-1::-1], conv_layer_sizes[-2::-1]
        ):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "conv": nn.ConvTranspose2d(
                            in_channels=layer_in[i["channels"]],
                            out_channels=layer_out[i["channels"]],
                            kernel_size=layer_in[i["kernel"]],
                            stride=layer_in[i["stride"]],
                            padding=layer_in[i["padding"]],
                        ),
                        "bn": nn.BatchNorm2d(layer_out[i["channels"]]),
                        "activation": activation_fn,
                    }
                )
            )

        self.blocks[-1]["activation"] = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        for block, conv_out_size in zip(self.blocks, self.conv_out_sizes[-2::-1]):
            x = block["conv"](
                x,
                output_size=torch.Size(
                    (
                        batch_size,
                        conv_out_size["channels"],
                        conv_out_size["width"],
                        conv_out_size["width"],
                    )
                ),
            )
            x = block["bn"](x)
            x = block["activation"](x)

        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        conv_dims: List = None,
        linear_dims: List = None,
        width: int = 28,
        in_channels: int = 1,
        latent_dim: int = 2,
    ):
        super().__init__()
        self.activation = nn.ReLU()

        if conv_dims is None:
            conv_dims = [[16, 3, 1, 0], [64, 3, 2, 0], [128, 5, 2, 0]]

        if linear_dims is None:
            linear_dims = [2048, 16]

        self.last_channel = conv_dims[-1][i["channels"]]

        # calculate the size of tensor after each convolutional layer
        conv_out_sizes = [{"channels": in_channels, "width": width}]

        conv_out = width
        for idx, dims in enumerate(conv_dims):
            conv_out = (
                conv_out + (2 * dims[i["padding"]]) - (dims[i["kernel"]] - 1) - 1
            ) / dims[i["stride"]] + 1
            conv_out = math.floor(conv_out)
            conv_out_sizes.append({"channels": dims[i["channels"]], "width": conv_out})

        self.conv_out = conv_out
        conv_dims.insert(0, [in_channels])

        linear_dims.insert(0, int(conv_out * conv_out * self.last_channel))
        linear_dims.append(latent_dim)

        self.encoder_conv = _ConvEncoder(conv_dims, self.activation)

        self.encoder_linear = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.05),
                )
                for dim_in, dim_out in zip(linear_dims, linear_dims[1::])
            ]
        )
        # replace final activation with tanh
        self.encoder_linear[-1][1] = nn.Tanh()

        self.decoder_linear = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.05),
                )
                for dim_in, dim_out in zip(linear_dims[-1::-1], linear_dims[-2::-1])
            ]
        )

        self.decoder_conv = _ConvDecoder(conv_dims, self.activation, conv_out_sizes)

    def encoder(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        h = self.encoder_linear(x)
        return h

    def decoder(self, h):
        x_hat = self.decoder_linear(h)
        x_hat = torch.reshape(
            x_hat, (-1, self.last_channel, self.conv_out, self.conv_out)
        )
        x_hat = self.decoder_conv(x_hat)
        return x_hat

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat


class VariationalAE(nn.Module):
    def __init__(
        self,
        conv_dims: List = None,
        linear_dims_encoder: List = None,
        width: int = 28,
        in_channels: int = 1,
        latent_dim: int = 2,
    ):
        super().__init__()
        self.activation = nn.ReLU()

        if conv_dims is None:
            conv_dims = [[16, 3, 1, 0], [64, 3, 2, 0], [128, 5, 2, 0]]

        if linear_dims_encoder is None:
            linear_dims_encoder = [[2048, 64], [64, 16]]


        self.last_channel = conv_dims[-1][i["channels"]]

        # calculate the size of tensor after each convolutional layer
        conv_out_sizes = [{"channels": in_channels, "width": width}]

        conv_out = width
        for idx, dims in enumerate(conv_dims):
            conv_out = (
                conv_out + (2 * dims[i["padding"]]) - (dims[i["kernel"]] - 1) - 1
            ) / dims[i["stride"]] + 1
            conv_out = math.floor(conv_out)
            conv_out_sizes.append({"channels": dims[i["channels"]], "width": conv_out})

        self.conv_out = conv_out
        conv_dims.insert(0, [in_channels])

        # the first list in linear_dims in the common fc network the second the individuals for mean and variance
        linear_dims_encoder[0].insert(0, int(conv_out * conv_out * self.last_channel))
        linear_dims_encoder[1].append(latent_dim)

        self.encoder_conv = _ConvEncoder(conv_dims, self.activation)

        self.encoder_linear_common = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.05),
                )
                for dim_in, dim_out in zip(linear_dims_encoder[0], linear_dims_encoder[0][1::])
            ]
        )

        self.encoder_linear_mean = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.05),
                )
                for dim_in, dim_out in zip(linear_dims_encoder[1], linear_dims_encoder[1][1::])
            ]
        )

        # restrict means in [-1, 1]
        self.encoder_linear_mean[-1][1] = nn.Tanh()

        self.encoder_linear_log_var = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.05),
                )
                for dim_in, dim_out in zip(linear_dims_encoder[1], linear_dims_encoder[1][1::])
            ]
        )

        linear_dims_decoder = linear_dims_encoder[0] + linear_dims_encoder[1]
        self.decoder_linear = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.05),
                )
                for dim_in, dim_out in zip(linear_dims_decoder[-1::-1], linear_dims_decoder[-2::-1])
            ]
        )

        self.decoder_conv = _ConvDecoder(conv_dims, self.activation, conv_out_sizes)

    def encoder(self, x):
        conv_out = self.encoder_conv(x)
        lin_common_out = self.encoder_linear_common(conv_out)

        mu = self.encoder_linear_mean(lin_common_out)
        log_var = self.encoder_linear_log_var(lin_common_out)

        return mu, log_var

    def reparametrize(self, mu, log_var):
        # sigma = e^(log(sigma^2)/2) = e^(log(sigma)) = sigma
        sigma = torch.exp(log_var/2)
        # sample from normal distribution
        eps = torch.randn_like(sigma)
        z = mu + sigma
        return z

    def decoder(self,z):
        lin_out = self.decoder_linear(z)
        x_hat = self.decoder_conv(lin_out)
        return x_hat

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, z, mu, log_var


if __name__ == "__main__":
    x = torch.rand(1, 1, 28, 28)
    conv_dims = [[4, 3, 1, 0], [16, 3, 1, 0], [32, 5, 1, 0]]

    in_channels = 1
    width = 28
    # calculate the size of tensor after each convolutional layer
    conv_out_sizes = [{"channels": in_channels, "width": width}]

    conv_out = width
    for idx, dims in enumerate(conv_dims):
        conv_out = (
            conv_out + (2 * dims[i["padding"]]) - (dims[i["kernel"]] - 1) - 1
        ) / dims[i["stride"]] + 1
        conv_out = math.floor(conv_out)
        conv_out_sizes.append({"channels": dims[i["channels"]], "width": conv_out})

    conv_dims.insert(0, [in_channels])

    m = _ConvEncoder(conv_dims, nn.ReLU())
    mt = _ConvDecoder(conv_dims, nn.ReLU(), conv_out_sizes)
    y = m(x)
    print(y.shape)

    k = mt(y)
    print(k.shape)
