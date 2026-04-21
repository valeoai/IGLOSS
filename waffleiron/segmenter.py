import torch.nn as nn

from .backbone import WaffleIron
from .embedding import Embedding


class Segmenter(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,
        grid_shape,
        drop_path_prob=0,
        gelu=False,
        mlp_classif=False,
        mlp_hidden_size=None,
        checkpointing=False,
    ):
        super().__init__()

        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)

        # WaffleIron backbone
        self.waffleiron = WaffleIron(
            feat_channels,
            depth,
            grid_shape,
            drop_path_prob,
            gelu,
            checkpointing,
        )

        # Classification layer
        if mlp_classif:
            self.classif = nn.Sequential(
                nn.Conv1d(feat_channels, mlp_hidden_size, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(mlp_hidden_size, nb_class, 1),
            )
        else:
            self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def forward(self, feats, cell_ind, occupied_cell, neighbors):
        tokens = self.embed(feats, neighbors)
        tokens = self.waffleiron(tokens, cell_ind, occupied_cell)
        return self.classif(tokens)
