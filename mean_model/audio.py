import torch
import torch.nn as nn


class MeanConcatDenseAudio(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.average_pool = nn.AvgPool1d(96)
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # average_pool = self.average_pool(audio_feat.permute(0,2,1)).squeeze(2)
        audio_emb = audio_feat.mean(1)
        # print('audio_emb', audio_emb)
        # print('delta', average_pool-audio_emb)
        audio_emb = self.audio_embed(audio_emb)
        output = self.outputLayer(audio_emb)
        return output

