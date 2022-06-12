import torch
import torch.nn as nn
from .common import AudioAttNet1, VideoAttNet1


class FreqAttention(nn.Module):
    """ Frequency Attention. """

    def __init__(self, device='cpu'):
        super(FreqAttention, self).__init__()
        self.h = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, bias=False),
            nn.ReLU(True),
        )
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        self.freq_attn = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):

        freq_attn = self.freq_attn(dct_x)
        # [32, 1, 299, 299] * [32, 3, 299, 299]
        return freq_attn * self.h(x)


class MeanConcatDenseDecisionMidAttn(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes, audio_attn=False, video_attn=False):
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
        self.outputLayer_audio = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.outputLayer_video = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.audio_attn = AudioAttNet1(256, 256)
        self.video_attn = VideoAttNet1(256, 256)
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)  # [b_s, 512]
        audio_emb = self.audio_embed(audio_emb)  # [b_s, 256]
        # audio_emb + video_emb_attn

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        # video_emb + audio_emb_attn

        audio_attn = self.audio_attn(audio_emb, video_emb)  # [128, 256]
        video_attn = self.video_attn(audio_emb, video_emb)
        # print('audio_attn', audio_attn.shape)

        video_emb = video_emb + audio_attn
        audio_emb = audio_emb + video_attn

        audio_output = self.outputLayer_audio(audio_emb)
        video_output = self.outputLayer_video(video_emb)
        output = audio_output + video_output
        return output

