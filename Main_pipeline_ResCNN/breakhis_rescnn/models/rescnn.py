import torch
import torch.nn as nn


class ResConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.shortcut = nn.Conv2d(ch_in, ch_out, 1, bias=False) if ch_in != ch_out else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.relu(out)


class ResCNNEncoderMLP(nn.Module):
    """ResCNN encoder with magnification embedding and two MLP heads.

    head1: benign/malignant logits, shape [B, 2]
    head2: 8-subtype logits, shape [B, 8]
    """

    def __init__(self, img_ch: int = 3, first_layer_numKernel: int = 64):
        super().__init__()
        k = first_layer_numKernel
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ResConvBlock(ch_in=img_ch, ch_out=k)
        self.Conv2 = ResConvBlock(ch_in=k, ch_out=2 * k)
        self.Conv3 = ResConvBlock(ch_in=2 * k, ch_out=4 * k)
        self.Conv4 = ResConvBlock(ch_in=4 * k, ch_out=8 * k)
        self.Conv5 = ResConvBlock(ch_in=8 * k, ch_out=16 * k)
        self.dropout_encoder = nn.Dropout2d(p=0.1)

        self.magn_embedding = nn.Embedding(4, 16 * k)

        self.head1 = nn.Sequential(nn.Linear(16 * k, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        self.head2 = nn.Sequential(nn.Linear(16 * k, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 8))

    def encode(self, x, magn, mask_token: bool = False):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        x5 = self.dropout_encoder(x5)

        magn_token = self.magn_embedding(magn).unsqueeze(-1).unsqueeze(-1)
        if mask_token:
            magn_token = torch.zeros_like(magn_token)
        x5 = x5 + magn_token
        return torch.mean(x5, dim=[2, 3])

    def forward(self, x, magn, mask_token: bool = False):
        x_feat = self.encode(x, magn, mask_token=mask_token)
        out1 = self.head1(x_feat)
        out2 = self.head2(x_feat)
        return out1, out2


# Backward-compatible name from the original notebook.
ResCNN = ResCNNEncoderMLP
