import torch
import torch.nn as nn


# baseline: 0.806
# baseline BN_plus: 0.822
# baseline BN_plus + Relu in outputLayer:0.818
class MeanConcatDenseMid(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        
        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputLayer(embed)
        return output

