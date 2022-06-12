import torch
import torch.nn as nn


class MeanDTcomT(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(4*audio_emb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.2)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(4*video_emb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(p=0.2)
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.average_pool = nn.AvgPool1d(kernel_size=36, stride=30)

        # 还可以尝试划窗信息和全局信息综合起来
        # 36 30 + all 4*audio_emb_dim + no dropout -> 0.822 (4*audio_emb_dim - 1024 - 512 - 128)

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        # 全局信息
        audio_emb_all = audio_feat.mean(1)
        video_emb_all = video_feat.mean(1)
        # 通过average_pool实现划窗操作+mean操作
        audio_average_pool = self.average_pool(audio_feat.permute(0, 2, 1))  # [bs, f, t']->[bs, t', f]
        video_average_pool = self.average_pool(video_feat.permute(0, 2, 1))
        # Flatten
        audio_emb_window = audio_average_pool.flatten(1)
        video_emb_window = video_average_pool.flatten(1)
        # 划窗信息和全局信息综合起来
        audio_emb = torch.cat((audio_emb_all, audio_emb_window), 1)
        video_emb = torch.cat((video_emb_all, video_emb_window), 1)
        # 早期特征建模
        audio_emb = self.audio_embed(audio_emb)
        video_emb = self.video_embed(video_emb)
        # 多模态融合
        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputLayer(embed)
        return output