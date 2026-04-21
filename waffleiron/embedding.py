import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        #
        self.compressed = False
        self.channels_in, self.channels_out = channels_in, channels_out

        # Normalize inputs
        self.norm = nn.BatchNorm1d(channels_in)

        # Point Embedding
        self.conv1 = nn.Conv1d(channels_in, channels_out, 1)

        # Neighborhood embedding
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 1, bias=False),
        )

        # Merge point and neighborhood embeddings
        self.final = nn.Conv1d(2 * channels_out, channels_out, 1, bias=True, padding=0)

    def forward(self, x, neighbors):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""
        # Normalize input
        x = self.norm(x)

        # Point embedding
        point_emb = self.conv1(x)

        # Neighborhood embedding
        gather = []
        # Gather neighbors around each center point
        for ind_nn in range(
            1, neighbors.shape[1]
        ):  # Remove first neighbors which is the center point
            temp = neighbors[:, ind_nn : ind_nn + 1, :].expand(-1, x.shape[1], -1)
            gather.append(torch.gather(x, 2, temp).unsqueeze(-1))
        # Relative coordinates
        neigh_emb = torch.cat(gather, -1) - x.unsqueeze(-1)  # Size: (B x C x N) x K
        # Embedding
        neigh_emb = self.conv2(neigh_emb).max(-1)[0]

        # Merge both embeddings
        return self.final(torch.cat((point_emb, neigh_emb), dim=1))
