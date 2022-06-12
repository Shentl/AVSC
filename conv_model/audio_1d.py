import torch
import torch.nn as nn


class ConvConcatDenseAudio1d(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        # in_channels, out_channels, kernel_size, stride, padding
        self.audio_embed = nn.Sequential(
            nn.Conv2d(1, 14, 5, 2, 2),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(14, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(28, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(28, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #
        )

        self.outputLayer = nn.Sequential(
            nn.Linear(28*48, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        audio_feat = audio_feat.unsqueeze(1)  # [128, 1, 96, 512]
        audio_emb = self.audio_embed(audio_feat)  # [128, 28, 12, 64] [bs, c, t/8, f/8]
        audio_emb = audio_emb.flatten(1)  # [128, 28, 48]
        # print('audio_emb', audio_emb.shape)
        output = self.outputLayer(audio_emb)
        print('output', output.shape)
        return output

