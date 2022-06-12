import torch
import torch.nn as nn


class MeanConcatDense(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes, mode='mid') -> None:
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.audio_embed_single = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.video_embed_single = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.early_embed = nn.Sequential(
            nn.Linear(video_emb_dim+audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        # 定义self.outputLayer_audio/video是用于decision_fusion
        self.outputLayer_audio = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )

        self.outputLayer_video = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )

    def single_audio(self, audio_feat):
        audio_emb = audio_feat.mean(1)  # [128, 512]
        audio_emb = self.audio_embed_single(audio_emb)  # [128, 256]
        audio_out = self.outputLayer_audio(audio_emb)  # [128, 10]
        return audio_out

    def single_video(self, video_feat):
        video_emb = video_feat.mean(1)  # [128, 512]
        video_emb = self.video_embed_single(video_emb)  # [128, 256]
        video_out = self.outputLayer_video(video_emb)  # [128, 10]
        return video_out

    def early_fusion(self, audio_feat, video_feat):
        audio_emb = audio_feat.mean(1)  # [128, 512]
        video_emb = video_feat.mean(1)
        av_emb = torch.cat((audio_emb, video_emb), 1)  # [128, 1024]
        embed = self.early_embed(av_emb)  # [128, 256]
        output = self.outputLayer(embed)  # [128, 10]
        return output

    def mid_term_fusion(self, audio_feat, video_feat):
        audio_emb = audio_feat.mean(1)  # [128, 512]
        audio_emb = self.audio_embed(audio_emb)  # [128, 128]
        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)  # [128, 128]

        embed = torch.cat((audio_emb, video_emb), 1)  # [128, 256]
        output = self.outputLayer(embed)  # [128, 10]
        return output

    def decision_fusion(self, audio_feat, video_feat):
        audio_out = self.single_audio(audio_feat)
        video_out = self.single_video(video_feat)

        output = audio_out + video_out  # [128, 10]
        return output

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        # video_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        # mode = 'mid' is the baseline
        if self.mode == 'early':
            output = self.early_fusion(audio_feat, video_feat)
        elif self.mode == 'mid':
            output = self.mid_term_fusion(audio_feat, video_feat)
        elif self.mode == 'decision':
            output = self.decision_fusion(audio_feat, video_feat)
        elif self.mode == 'audio':
            output = self.single_audio(audio_feat)
        else:
            output = self.single_audio(video_feat)
        return output

