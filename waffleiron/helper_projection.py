import torch


def projection_3d_to_2d_scatter_reduce(feat, sp_mat, B, C, H, W):

    residual = torch.zeros((B, C, H * W), device=feat.device, dtype=feat.dtype)
    residual.scatter_reduce_(
        2,
        sp_mat["inflate"],
        feat,
        "mean",
        include_self=False,
    )

    return residual


def get_all_projections_scatter_reduce(cell_ind, nb_feat, *args, **kwargs):
    sp_mat = [
        {"inflate": cell_ind[:, i : i + 1].expand(-1, nb_feat, -1)}
        for i in range(cell_ind.shape[1])
    ]
    return sp_mat
