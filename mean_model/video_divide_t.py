import torch
import torch.nn as nn


class MeanVDivideT(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.video_embed = nn.Sequential(
            nn.Linear(3*video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
            # 加了这些后划窗有提升, 但加在纯video里无提升, 同时video里最后加softmax有反作用
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
        self.average_pool = nn.AvgPool1d(kernel_size=48, stride=24)
        # self.average_pool = nn.AvgPool1d(kernel_size=96)

    def forward(self, audio_feat, video_feat):
        # video_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        video_average_pool = self.average_pool(video_feat.permute(0, 2, 1)).permute(0, 2, 1)
        # print('video_average_pool', video_average_pool.shape)  # [128, 512, 5]
        video_emb = video_average_pool.flatten(1)
        # print(video_emb.shape)
        video_emb = self.video_embed(video_emb)
        output = self.outputLayer(video_emb)
        # output = torch.sigmoid(output)  # [128,10]
        return output

