import torch
import torch.nn as nn


class DevideTimeMean(nn.Module):
    # 尝试将t分块做mean，然后flatten，而不是直接就整体mean了
    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
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
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.average_pool = nn.AvgPool1d(kernel_size=16, stride=8)

    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        # video_feat: [batch_size, time_steps, feat_dim]
        # 可以用pooling来做分块mean
        audio_average_pool = self.average_pool(audio_feat.permute(0,2,1))
        audio_emb = audio_feat.mean(1)
        print('audio_emb', audio_emb.shape)
        print('audio_average_pool', audio_average_pool.shape)
        exit()
        """

        # pool with window of size=3, stride=2
        m = nn.AvgPool1d(3, stride=2)
        m(torch.tensor([[[1.,2,3,4,5,6,7]]]))
        tensor([[[ 2.,  4.,  6.]]])
        """
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)

        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputLayer(embed)
        return output