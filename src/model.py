import torch
import torch.nn as nn

from torchinfo import summary

class EncoderBranch(nn.Module):
    def __init__(self, num_heads=4):
        super(EncoderBranch, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 512, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, 3, stride=(2,1), padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, stride=(2,1), padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.attention = nn.MultiheadAttention(32, num_heads, batch_first=True, bias=True)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1, x.size(1))
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        return x

class Encoder(nn.Module):
    def __init__(self, z_dim=128, num_heads=4):
        super(Encoder, self).__init__()

        self.encoder_branch1 = EncoderBranch(num_heads=num_heads)
        self.encoder_branch2 = EncoderBranch(num_heads=num_heads)
        self.encoder_branch3 = EncoderBranch(num_heads=num_heads)

        self.attention = nn.MultiheadAttention(32*3, num_heads, batch_first=True, bias=True)
        self.fc = nn.Linear(32 * 3 * 4 * 4, z_dim)

    def forward(self, x1, x2, x3):
        x1 = self.encoder_branch1(x1)
        x2 = self.encoder_branch2(x2)
        x3 = self.encoder_branch3(x3)

        x = torch.cat((x1, x2, x3), dim=2)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim=128):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(z_dim, 32*4*8)

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=(2,1), padding=1, output_padding=(1,0), bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 128, 3, stride=(2,1), padding=1, output_padding=(1,0), bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 256, 3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 512, 3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, 1, 3, stride=2, padding=1, output_padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.size(0), 32, 4, 8)
        x = self.cnn(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, z_dim=128):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

    def forward(self, x1, x2, x3):
        x = self.encoder(x1, x2, x3)
        x = self.decoder(x)
        return x


if __name__=='__main__':
    model = Encoder(z_dim=128)
    summary(model, input_size=((64,1,128,32), (64,1,128,32), (64,1,128,32)), col_names=["input_size", "output_size"])