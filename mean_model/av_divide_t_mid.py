import torch
import torch.nn as nn


class MeanAVDivideTMid(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(2*audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(2*video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.average_pool = nn.AvgPool1d(kernel_size=72, stride=24)
        # 还可以尝试划窗信息和全局信息综合起来
        # baseline(mid) 0.806
        # 72 24 2*audio_emb_dim -> 0.814
        # 36 30 3*audio_emb_dim -> 0.806
        # 48 24 3*audio_emb_dim -> 0.810 flatten前permute 0.806
        # 36,20 4*audio_emb_dim-512-128 -> 0.814 flatten前permute 0.813
        # 32 16 5*audio_emb_dim -> 0.804 flatten前permute 0.806
        # 32 16 5*audio_emb_dim-1024-256-128 -> 0.795 flatten前permute 0.791，但training中的cv acc高，有0.85
        # 上参 5*audio_emb_dim-1024-512-128 -> 0.807 flatten前permute 0.807

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_average_pool = self.average_pool(audio_feat.permute(0, 2, 1))  # [bs, f, t']->[bs, t', f]
        # .permute(0, 2, 1)
        video_average_pool = self.average_pool(video_feat.permute(0, 2, 1))
        # .permute(0, 2, 1)
        audio_emb = audio_average_pool.flatten(1)
        video_emb = video_average_pool.flatten(1)

        audio_emb = self.audio_embed(audio_emb)
        video_emb = self.video_embed(video_emb)

        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputLayer(embed)
        return output
