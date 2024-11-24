import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DeepPixBiS_DualBranch_Shared(nn.Module):
    """
    Dual-Branch DeepPixBiS Model with Shared Weights.
    """

    def __init__(self, pretrained=True):
        super(DeepPixBiS_DualBranch_Shared, self).__init__()

        # Shared encoder and decoder
        dense = densenet121(weights=DenseNet121_Weights.DEFAULT)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[0:8])
        self.dec = nn.Conv2d(256, 1, kernel_size=1, padding=0)

        # Fully connected layer
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x_face, x_context):
        # Process face input
        enc_face = self.enc(x_face)
        dec_face = self.dec(enc_face)
        dec_face = torch.sigmoid(dec_face)

        # Process context input
        enc_context = self.enc(x_context)
        dec_context = self.dec(enc_context)
        dec_context = torch.sigmoid(dec_context)
        dec_context_flat = dec_context.view(-1, 14 * 14)

        # fc & sigmoid
        binary_out = self.linear(dec_context_flat)
        binary_out = torch.sigmoid(binary_out)


        return dec_face, dec_context, binary_out
    