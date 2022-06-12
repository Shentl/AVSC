import torch
import torch.nn as nn


class MeanADivideT(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(3*audio_emb_dim, 512),
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
        audio_average_pool = self.average_pool(audio_feat.permute(0, 2, 1)).permute(0, 2, 1)
        # print('video_average_pool', video_average_pool.shape)  # [128, 512, 5]
        audio_emb = audio_average_pool.flatten(1)
        # print(video_emb.shape)
        audio_emb = self.audio_embed(audio_emb)
        output = self.outputLayer(audio_emb)
        return output
