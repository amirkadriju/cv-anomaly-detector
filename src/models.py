# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Autoencoder ----
class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[16, 32, 64]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[2], hidden_dims[1], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[0], in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        x_reconstructed = F.interpolate(x_reconstructed, size=x.size()[2:], mode='bilinear', align_corners=False)
        return x_reconstructed


# ---- Classifier ----
class AnomalyClassifier(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))


# ---- UNet ----
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(),
            )
        self.encoder1 = conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_filters * 2, base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.decoder2 = conv_block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.decoder1 = conv_block(base_filters * 2, base_filters)

        self.final = nn.Conv2d(base_filters, out_channels, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# === Model Factory ===
def get_model(model_name: str, config: dict) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 1),
            base_filters=config.get("base_filters", 64)
        )
    elif model_name == "autoencoder":
        return Autoencoder(
            in_channels=config.get("in_channels", 3),
            hidden_dims=config.get("hidden_dims", [16, 32, 64])
        )
    elif model_name == "classifier":
        return AnomalyClassifier(
            in_channels=config.get("in_channels", 3),
            hidden_dims=config.get("hidden_dims", [32, 64])
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
