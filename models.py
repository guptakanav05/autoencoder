import torch
import torch.nn as nn


class FC_Autoencoder_v1(nn.Module):

    def __init__(self):
        super(FC_Autoencoder_v1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(batch_size, 1, 28, 28)
        return reconstructed

    def encode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.encoder(x)


class FC_Autoencoder_v2(nn.Module):

    def __init__(self):
        super(FC_Autoencoder_v2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(batch_size, 1, 28, 28)
        return reconstructed

    def encode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.encoder(x)


class CNN_Autoencoder_v1(nn.Module):

    def __init__(self):
        super(CNN_Autoencoder_v1, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 32),
            nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(32, 32 * 7 * 7),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder_conv(x)
        x = x.view(batch_size, -1)
        latent = self.encoder_fc(x)
        x = self.decoder_fc(latent)
        x = x.view(batch_size, 32, 7, 7)
        reconstructed = self.decoder_conv(x)
        return reconstructed

    def encode(self, x):
        batch_size = x.size(0)
        x = self.encoder_conv(x)
        x = x.view(batch_size, -1)
        return self.encoder_fc(x)


class CNN_Autoencoder_v2(nn.Module):

    def __init__(self):
        super(CNN_Autoencoder_v2, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 64),
            nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(64, 128 * 7 * 7),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder_conv(x)
        x = x.view(batch_size, -1)
        latent = self.encoder_fc(x)
        x = self.decoder_fc(latent)
        x = x.view(batch_size, 128, 7, 7)
        reconstructed = self.decoder_conv(x)
        return reconstructed

    def encode(self, x):
        batch_size = x.size(0)
        x = self.encoder_conv(x)
        x = x.view(batch_size, -1)
        return self.encoder_fc(x)


def get_model(model_name):
    models = {
        'fc_v1': FC_Autoencoder_v1,
        'fc_v2': FC_Autoencoder_v2,
        'cnn_v1': CNN_Autoencoder_v1,
        'cnn_v2': CNN_Autoencoder_v2,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()


MODEL_INFO = {
    'fc_v1': {
        'name': 'FC Autoencoder v1',
        'type': 'Fully Connected',
        'bottleneck': 32,
        'description': 'Simple FC autoencoder (784→256→128→32→128→256→784)'
    },
    'fc_v2': {
        'name': 'FC Autoencoder v2',
        'type': 'Fully Connected',
        'bottleneck': 64,
        'description': 'Deeper FC with BatchNorm (784→512→256→128→64→128→256→512→784)'
    },
    'cnn_v1': {
        'name': 'CNN Autoencoder v1',
        'type': 'Convolutional',
        'bottleneck': 32,
        'description': 'Simple CNN autoencoder (Conv16→Conv32→FC32→ConvT16→ConvT1)'
    },
    'cnn_v2': {
        'name': 'CNN Autoencoder v2',
        'type': 'Convolutional',
        'bottleneck': 64,
        'description': 'Deeper CNN with BatchNorm (Conv32→Conv64→Conv128→FC64→ConvT64→ConvT32→ConvT1)'
    }
}


if __name__ == '__main__':
    print("Testing all models with a dummy batch...\n")
    dummy_input = torch.randn(4, 1, 28, 28)

    for name, info in MODEL_INFO.items():
        model = get_model(name)
        output = model(dummy_input)
        latent = model.encode(dummy_input)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Parameters:   {total_params:,}")
        print()
