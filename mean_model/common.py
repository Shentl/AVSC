import torch
import torch.nn as nn


class AudioAttNet(nn.Module):
    """
    Attention from audio to video
    input: [b_s, 256] or [batch_size, time_steps, feat_dim]
    """
    def __init__(self, dim_aud=256, dim_video=256):
        super(AudioAttNet, self).__init__()
        self.dim_aud = dim_aud
        self.dim_video = dim_video
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.dim_aud, out_features=self.dim_video),
            nn.Softmax(dim=1)
        )

    def forward(self, audio_emb, video_emb):
        # [b_s, 256]
        y = self.attentionNet(audio_emb)  # [128, 256]
        # print('y', y.shape)
        attn = y*video_emb  # [128, 256]
        # print('attn', attn.shape)
        return attn


class VideoAttNet(nn.Module):
    """
    Attention from video to audio
    input: [b_s, 256] or [batch_size, time_steps, feat_dim]
    """
    def __init__(self, dim_aud=256, dim_video=256):
        super(VideoAttNet, self).__init__()
        self.dim_aud = dim_aud
        self.dim_video = dim_video
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.dim_video, out_features=self.dim_aud),
            nn.Softmax(dim=1)
        )

    def forward(self, audio_emb, video_emb):
        # [b_s, 256]
        y = self.attentionNet(video_emb)  # [128, 256]
        # print('y', y.shape)
        attn = y*audio_emb  # [128, 256]
        # print('attn', attn.shape)
        return attn


class AudioAttNet1(nn.Module):
    """
    Attention from audio to video, apply on video
    input: [b_s, 256] or [batch_size, time_steps, feat_dim]
    """
    def __init__(self, dim_aud=256, dim_video=256):
        super(AudioAttNet1, self).__init__()
        self.dim_aud = dim_aud
        self.dim_video = dim_video
        self.audioNet = nn.Linear(in_features=self.dim_aud, out_features=self.dim_aud)
        self.videoNet1 = nn.Linear(in_features=self.dim_video, out_features=self.dim_video)
        self.videoNet2 = nn.Linear(in_features=self.dim_video, out_features=self.dim_video)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, audio_emb, video_emb):
        # [b_s, 256]
        y_aud = self.audioNet(audio_emb)  # [128, 256]
        y_video = self.videoNet1(video_emb)
        we = y_aud*y_video  # [128, 256]
        we = self.softmax(we)
        x_video = self.videoNet2(video_emb)  # [128, 256]

        return we*x_video


class VideoAttNet1(nn.Module):
    """
    Attention from video to audio, apply on audio
    input: [b_s, 256] or [batch_size, time_steps, feat_dim]
    """
    def __init__(self, dim_aud=256, dim_video=256):
        super(VideoAttNet1, self).__init__()
        self.dim_aud = dim_aud
        self.dim_video = dim_video
        self.audioNet1 = nn.Linear(in_features=self.dim_aud, out_features=self.dim_aud)
        self.audioNet2 = nn.Linear(in_features=self.dim_aud, out_features=self.dim_aud)
        self.videoNet = nn.Linear(in_features=self.dim_video, out_features=self.dim_video)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, audio_emb, video_emb):
        # [b_s, 256]
        y_aud = self.audioNet1(audio_emb)  # [128, 256]
        y_video = self.videoNet(video_emb)
        we = y_aud * y_video  # [128, 256]
        we = self.softmax(we)
        x_aud = self.audioNet2(audio_emb)  # [128, 256]

        return we*x_aud


