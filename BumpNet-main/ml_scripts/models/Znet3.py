import torch
import torch.nn as nn


def build_conv_seq(kernels, channels):
    layers = []
    for i, kern in enumerate(kernels):
        layers.append(
            nn.Conv1d(
                in_channels = channels[i], out_channels = channels[i+1],
                kernel_size = kern, padding = 'same'
            )
        )
        layers.append(nn.ReLU())
    seq = nn.Sequential(*layers)
    return seq



class Znet(nn.Module):

    def __init__(self, scale=None):
        super().__init__()

        if scale is None:
            scale = {
                'bin_content': {'min': 0.0, 'max': 1.0},
                'true_z': {'min': 0.0, 'max': 1.0},
                'background': {'min': 0.0, 'max': 1.0}
            }

        # normalization
        self.min_x = torch.nn.Parameter(torch.FloatTensor([scale['bin_content']['min']]), requires_grad=False)
        self.max_x = torch.nn.Parameter(torch.FloatTensor([scale['bin_content']['max']]), requires_grad=False)

        self.min_z = torch.nn.Parameter(torch.FloatTensor([scale['true_z']['min']]), requires_grad=False)
        self.max_z = torch.nn.Parameter(torch.FloatTensor([scale['true_z']['max']]), requires_grad=False)
        
        self.min_b = torch.nn.Parameter(torch.FloatTensor([scale['background']['min']]), requires_grad=False)
        self.max_b = torch.nn.Parameter(torch.FloatTensor([scale['background']['max']]), requires_grad=False)



        self.kernels = [
            [25, 25, 25, 25, 25],
            [15, 15, 15, 15, 15],
            [ 9,  9,  9,  9,  9],
            [ 3,  3,  3,  3,  3]
        ]

        self.channels = [
            [1, 64, 64, 64, 64, 64],
            [1, 64, 64, 64, 64, 64],
            [1, 64, 64, 64, 64, 64],
            [1, 64, 64, 64, 64, 64]
        ]



        self.conv_seq1 = build_conv_seq(self.kernels[0], self.channels[0])
        self.conv_seq2 = build_conv_seq(self.kernels[1], self.channels[1])
        self.conv_seq3 = build_conv_seq(self.kernels[2], self.channels[2])
        self.conv_seq4 = build_conv_seq(self.kernels[3], self.channels[3])

        conv_output_ch = sum([1] + [x[-1] for x in self.channels])
        linear_layers = [conv_output_ch, 128, 64, 32, 1]

        dense_seq_z = []
        dense_seq_b = []

        for i in range(len(linear_layers) - 2):
            dense_seq_z.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            dense_seq_b.append(nn.Linear(linear_layers[i], linear_layers[i+1]))

            dense_seq_z.append(nn.ReLU())
            dense_seq_b.append(nn.ReLU())

        dense_seq_z.append(nn.Linear(linear_layers[-2], linear_layers[-1]))
        dense_seq_b.append(nn.Linear(linear_layers[-2], linear_layers[-1]))

        self.dense_seq_z = nn.Sequential(*dense_seq_z)    
        self.dense_seq_b = nn.Sequential(*dense_seq_b)    



    def forward(self, x):

        # normalize input
        x = (x - self.min_x) / (self.max_x - self.min_x)

        b, n = x.shape
        x = x.reshape(b, 1, n) # conv1d expects (B, C, N)

        outputs = []
        outputs.append(self.conv_seq1(x))
        outputs.append(self.conv_seq2(x))
        outputs.append(self.conv_seq3(x))
        outputs.append(self.conv_seq4(x))

        outputs = torch.cat(outputs + [x], dim=1)
        outputs = outputs.transpose(1,2)  # converting it to (B, N, C)

        z_pred = self.dense_seq_z(outputs)
        b_pred = self.dense_seq_b(outputs)

        z_pred = z_pred.reshape(b, n)
        b_pred = b_pred.reshape(b, n)


        # undo normalize the output
        z_pred = z_pred * (self.max_z - self.min_z) + self.min_z
        b_pred = b_pred * (self.max_b - self.min_b) + self.min_b

        return z_pred, b_pred
    