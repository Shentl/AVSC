import torch
import torch.nn as nn

# Integrating the Data Augmentation Scheme with Various Classifiers for Acoustic Scene Modeling
# audio_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
# mean_model: audio_emb = audio_feat.mean(1) [batch_size, 1, feat_dim]
#  self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)), 降freq不降time
class ConvConcatDenseAudio(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        # in_channels, out_channels, kernel_size, stride, padding
        self.audio_embed = nn.Sequential(
            nn.Conv2d(1, 14, 5, 2, 2),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Conv2d(14, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv1 [128, 28, 24, 128]

            nn.Conv2d(28, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 28, 3, 1, 1),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv2 [128, 28, 12, 64]

            nn.Conv2d(28, 56, 3, 1, 1),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(56, 56, 3, 1, 1),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(56, 56, 3, 1, 1),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(56, 56, 3, 1, 1),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv3 ([128, 56, 6, 32]

            nn.Conv2d(56, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # conv4 ([128, 128, 6, 32]

            nn.Conv2d(128, 10, 1, 1, 0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # [128, 10, 6, 32]
            nn.AdaptiveAvgPool2d((1, 1))
            # [128, 10, 1, 1]
            # 把channnel降到10，剩下的feature直接global_averge_pool到1维

        )

        self.outputLayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim] [128, 96, 512]
        audio_feat = audio_feat.unsqueeze(1)  # [128, 1, 96, 512]
        audio_emb = self.audio_embed(audio_feat)  # [1, 10, 1, 1]
        # print('audio_emb', audio_emb.shape)
        output = audio_emb.squeeze(dim=2).squeeze(dim=2)
        return output

