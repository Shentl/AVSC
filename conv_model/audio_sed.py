import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    # [batch_size, channel, time_steps, feat_dim]
    def __init__(self, in_channels=1, out_channels=1, t_ksize=3, f_ksize=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(t_ksize, f_ksize),
                              stride=1, padding=((t_ksize-1)//2, (f_ksize-1)//2))
        self.BN = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        out = F.relu(self.BN(self.conv(x)))
        out = self.pool(out)
        return out


class ConvAudioSed(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super(ConvAudioSed, self).__init__()
        self.num_classes = num_classes
        # in_channels, out_channels, kernel_size, stride, padding
        self.b1 = ConvBlock(in_channels=1, out_channels=16, t_ksize=9, f_ksize=9)
        self.block = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=16, t_ksize=9, f_ksize=9),
            ConvBlock(in_channels=16, out_channels=32, t_ksize=9, f_ksize=7),
            ConvBlock(in_channels=32, out_channels=64, t_ksize=7, f_ksize=7),
            ConvBlock(in_channels=64, out_channels=128, t_ksize=7, f_ksize=7),
            ConvBlock(in_channels=128, out_channels=128, t_ksize=7, f_ksize=7)
        )
        # [b_s, c, T/8, F/8]
        self.fc = nn.Linear(2 * 512, self.num_classes)
        # self.fc2 = nn.Linear(128, class_num)
        self.rnn = nn.GRU(input_size=16 * 128, hidden_size=512, num_layers=3, batch_first=True, dropout=0,
                          bidirectional=True)

    def forward(self, audio_feat, video_feat):
        x = audio_feat.unsqueeze(dim=1)
        # x [128, 1, 96, 512] [batch_size, 1, time_steps, feat_dim]
        out = self.block(x)  # [128, 256, 96, 16] [batch_size, c=256, time_steps, feat_dim/32]
        # using stack
        out = out.permute(0, 2, 3, 1).flatten(2)  # stack at time_dim [128, 96, 16, 256] [b_s, t, f/32, c]
        # [128, 96, 2048]
        out, hidden = self.rnn(out)  # [32, 501, 128*2] [b_s, t, 2c]
        out = self.fc(out)  # [32, 501, 10] [b_s, T, class_num]
        print('out', out.shape)
        return out
