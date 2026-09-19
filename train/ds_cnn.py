from collections import OrderedDict

import torch
from torch import nn

N_FRAMES, N_MFCC = 49, 10


def conv_bn_relu(cin, cout, kernel, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


def layer_names(blocks):
    return ["conv1"] + [f"{kind}{i}" for i in range(1, blocks + 1) for kind in ("dw", "pw")]


class DSCNN(nn.Module):
    """depthwise separable cnn keyword spotter, ds-cnn-s layout from hello edge.

    input (n, 1, 49, 10) normalized mfcc, output (n, n_classes) logits.
    conv1 is 10x4 stride 2 down to 25x5, then each block is a 3x3 depthwise
    and a 1x1 pointwise conv, all followed by bn + relu. width 64/172/276
    matches the s/m/l channel counts of the arm ml-zoo models.
    """

    def __init__(self, n_classes=35, width=172, blocks=4, dropout=0.2):
        super().__init__()
        self.width, self.blocks, self.n_classes = width, blocks, n_classes
        layers = [("conv1", conv_bn_relu(1, width, (10, 4), stride=2, padding=(5, 1)))]
        for i in range(1, blocks + 1):
            layers.append((f"dw{i}", conv_bn_relu(width, width, 3, padding=1, groups=width)))
            layers.append((f"pw{i}", conv_bn_relu(width, width, 1)))
        self.features = nn.Sequential(OrderedDict(layers))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


def load(path, device="cpu"):
    """rebuild a model from a train.py checkpoint.

    returns (model in eval mode, checkpoint dict). the checkpoint also carries
    the feature normalization stats every runtime needs.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DSCNN(ckpt["n_classes"], ckpt["width"], ckpt["blocks"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
