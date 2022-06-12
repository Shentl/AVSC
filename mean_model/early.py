import torch
import torch.nn as nn


# early: 0.787
# early BN_plus: 0.801
class MeanConcatDenseEarly(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.early_embed = nn.Sequential(
            nn.Linear(video_emb_dim + audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)  # [128, 512]
        video_emb = video_feat.mean(1)
        av_emb = torch.cat((audio_emb, video_emb), 1)  # [128, 1024]
        embed = self.early_embed(av_emb)  # [128, 256]
        output = self.outputLayer(embed)  # [128, 10]
        return output
