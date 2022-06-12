import torch
import torch.nn as nn
from .common import VideoAttNet1


class MeanAudioVattn(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )

        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )

        self.video_attn = VideoAttNet1(256, 256)
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)  # [b_s, 256]

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)  # [b_s, 256]

        video_attn = self.video_attn(audio_emb, video_emb)
        audio_emb = audio_emb + video_attn

        output = self.outputLayer(audio_emb)
        return output

