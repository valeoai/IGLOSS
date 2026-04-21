import argparse
import os

import torch
import yaml
from datasets import LIST_DATASETS, Collate
from tqdm import tqdm
from waffleiron import Segmenter


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


def extract_features(gpu, args, config):

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
    for it, batch in enumerate(tqdm(loader)):
        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
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

        # Save
        save_features(
            out.float().cpu().numpy(),
            args.log_path,
            batch["filename"][0],
            args.dataset.lower(),
        )


def main(args, config):

    # --- Fixed args
    # Device
    args.device = "cuda"
    # Node rank for distributed training
    args.gpu = 0
    args.rank = 0
    # Number of nodes for distributed training'
    args.world_size = 1

    # Create log directory
    os.makedirs(args.log_path, exist_ok=True)

    # Extract features
    extract_features(args.gpu, args, config)


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
        "--log_path", type=str, required=True, help="Path to log folder"
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
    return parser


if __name__ == "__main__":
    # Get args
    parser = get_default_parser()
    args = parser.parse_args()

    # Load config files
    config = load_model_config(args.config)

    # Launch training
    main(args, config)
