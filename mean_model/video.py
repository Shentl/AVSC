import torch
import torch.nn as nn


class MeanConcatDenseVideo(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
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
        # video_feat: [batch_size, time_steps, feat_dim]
        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        output = self.outputLayer(video_emb)
        # output = torch.sigmoid(output)  # [128,10]
        return output

