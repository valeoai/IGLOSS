import argparse
import os
import pickle
import time
import numpy as np
import torch
import yaml
from datasets import LIST_DATASETS, Collate
from tqdm import tqdm
from waffleiron import Segmenter
 
from igloss_utils import (
    fast_hist,
    flatten_with_mapping,
    overall_accuracy,
    per_class_accuracy,
    per_class_iu,
    print_log,
    CLASS_NAMES_NUSCENES,
    CLASS_NAMES_SEMANTIC_KITTI
)
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_features(features, log_path, filename, dataset):

    if dataset == "nuscenes":
        pass
    elif dataset == "semantic_kitti":
        pass
    else:
        raise ValueError(f"No saving function for {dataset}")


def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_dataset(config, args, phase="train"):

    # Shared parameters
    kwargs = {
        "rootdir": args.path_dataset,
        "input_feat": config["point_backbone"]["input_features"],
        "voxel_size": config["point_backbone"]["voxel_size"],
        "num_neighbors": config["point_backbone"]["num_neighbors"],
        "dim_proj": config["point_backbone"]["dim_proj"],
        "grids_shape": config["point_backbone"]["grid_shape"],
        "fov_xyz": config["point_backbone"]["fov"],
        "force_upsample": True,
    }

    # Get datatset
    DATASET = LIST_DATASETS.get(args.dataset.lower())
    if DATASET is None:
        raise ValueError(f"Dataset {args.dataset.lower()} not available.")

    # Train dataset
    dataset = DATASET(
        phase=phase,
        **kwargs,
    )

    return dataset


def get_dataloader(dataset, args):

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )

    return dataset_loader


def predict_w_logistic_regression(
    loader, teacher, template_features, mapping_tensor, dataset
):

    confusion_matrix = 0
    print_freq = np.max((len(loader) // 20, 1))
    RANDOM_STATE = 0
    REG_LOGISTIC = 1
    template_features = torch.nn.functional.normalize(template_features, p=2, dim=1)
    template_features = template_features.cpu().data.numpy()
    
    clf = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="liblinear",
        C=REG_LOGISTIC,
    ).fit(template_features, mapping_tensor)

    if dataset == "semantic_kitti":
        class_num = len(CLASS_NAMES_SEMANTIC_KITTI)
    elif dataset == "nuscenes":
        class_num = len(CLASS_NAMES_NUSCENES)
    else:
        raise ValueError(f"{dataset} is not supported.")


    W = torch.from_numpy(clf.coef_)
    b = torch.from_numpy(clf.intercept_)
    mapping_tensor = torch.from_numpy(mapping_tensor)
    W, b, mapping_tensor = W.to(device), b.to(device), mapping_tensor.to(device)

    for it, batch in enumerate(tqdm(loader)):
        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
        labels = batch["labels_orig"].cuda(non_blocking=True)
        batch["upsample"] = [up.cuda(non_blocking=True) for up in batch["upsample"]]

        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction and loss
        with torch.autocast("cuda", enabled=args.fp16):
            with torch.no_grad():
                out = teacher(*net_inputs) 
                out_upsample = []
                for id_b, closest_point in enumerate(batch["upsample"]):
                    temp = out[id_b, :, closest_point]
                    out_upsample.append(temp.T)
                out = torch.cat(out_upsample, dim=0)

        out = out.float()

        valid_inds = labels != 255

        valid_feats = out[valid_inds, :]

        valid_feats_norm = torch.nn.functional.normalize(
            valid_feats.cuda(), p=2, dim=1
        ).to(torch.float64)

        logits = valid_feats_norm @ W.T + b
        y_prediction_i = torch.softmax(logits, dim=1).max(1)[1]

        confusion_matrix += fast_hist(y_prediction_i, labels[valid_inds], class_num)

        if it % print_freq == print_freq - 1 or it == len(loader) - 1:
            oAcc = 100 * overall_accuracy(confusion_matrix.cpu())
            mAcc = 100 * np.nanmean(per_class_accuracy(confusion_matrix.cpu()))
            ious = per_class_iu(confusion_matrix.cpu())
            mIoU = 100 * np.nanmean(ious)
            print_log(oAcc, mAcc, mIoU, ious, dataset)


def open_voc_segmentation_w_scalr_plus(gpu, args, config):

    # --- Init. distributing training
    args.gpu = gpu
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # --- Build network
    teacher = Segmenter(
        input_channels=config["point_backbone"]["size_input"],
        feat_channels=config["point_backbone"]["nb_channels"],
        nb_class=config["point_backbone"]["nb_class"],
        depth=config["point_backbone"]["depth"],
        grid_shape=config["point_backbone"]["grid_shape"],
        drop_path_prob=config["point_backbone"]["drop_path"],
        gelu=config["point_backbone"]["gelu"],
        mlp_classif=config["point_backbone"]["mlp_classif"],
        mlp_hidden_size=config["point_backbone"]["hidden"],
    )

    # --- Load pretrained model
    ckpt = torch.load(
        args.wi_pretrained_ckpt,
        map_location="cpu",
    )["model_point"]
    new_ckpt = {}
    for k in ckpt.keys():
        weight = ckpt[k]
        if k.startswith("module"):
            new_ckpt[k[len("module.") :]] = weight
        else:
            new_ckpt[k] = weight
    msg = teacher.load_state_dict(new_ckpt, strict=True)
    print(teacher)
    print(f"[{args.gpu}]", msg)

    # ---
    args.workers = config["dataloader"]["num_workers"]
    torch.cuda.set_device(args.gpu)
    teacher = teacher.cuda(args.gpu)

    # --- Freeze teacher parameters
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.eval()

    # --- Dataset
    dataset = get_dataset(config, args, phase=args.split)
    dataset.force_upsample = True
    loader = get_dataloader(dataset, args)

    # ---
    with open(args.templates,'rb') as f:
        template_features = pickle.load(f)

    template_features, mapping_tensor = flatten_with_mapping(template_features)

    predict_w_logistic_regression(loader, teacher, template_features, mapping_tensor, args.dataset)

def main(args, config):

    # --- Fixed args
    # Device
    args.device = "cuda"
    # Node rank for distributed training
    args.gpu = 0
    args.rank = 0
    # Number of nodes for distributed training'
    args.world_size = 1

    # Apply IGLOSS
    open_voc_segmentation_w_scalr_plus(args.gpu, args, config)


def get_default_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset",
        default="nuscenes",
    )
    parser.add_argument(
        "--path_dataset",
        type=str,
        help="Path to dataset",
        default="/datasets_local/nuscenes/",
    )
    parser.add_argument(
        "--log_path", type=str, required=False, help="Path to log folder"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config downstream",
    )
    parser.add_argument(
        "--wi_pretrained_ckpt",
        default="",
        type=str,
        required=True,
        help="Path to WI pretrained ckpt",
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        help="train or val",
    )
    parser.add_argument(
        "--templates",
        required=True,
        type=str,
        help="Path for template features",
    )
    return parser


if __name__ == "__main__":
    # Get args
    parser = get_default_parser()
    args = parser.parse_args()

    # Load config files
    config = load_model_config(args.config)

    # Launch training
    main(args, config)
