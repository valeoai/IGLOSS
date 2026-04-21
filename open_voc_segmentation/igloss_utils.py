import sys

import numpy as np
import torch

CLASS_NAMES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASS_NAMES_SEMANTIC_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

def flatten_with_mapping(nested_list):
    flat_values = []
    mapping = []

    for group_idx, sublist in enumerate(nested_list):
        # Convert sublist to 1D tensor if it's a list

        for sub_tensor in sublist:
            flat_values.append(sub_tensor)
            mapping.append(torch.full((len(sub_tensor),), group_idx, dtype=torch.long))

    # Concatenate the flattened tensor and the group mapping
    flat_tensor = torch.cat(flat_values) if flat_values else torch.tensor([])
    mapping_tensor = (
        torch.cat(mapping) if mapping else torch.tensor([], dtype=torch.long)
    )

    return flat_tensor, mapping_tensor.cpu().data.numpy()


def fast_hist(pred, label, n):
    assert torch.all(label > -1) & torch.all(pred > -1)
    assert torch.all(label < n) & torch.all(pred < n)
    return torch.bincount(n * label + pred, minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def overall_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist).sum() / hist.sum()


def per_class_accuracy(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / hist.sum(1)


def print_log(oAcc, mAcc, mIoU, ious, dataset):
    # Global score

    if dataset=="nuscenes":
        CLASS_NAMES = CLASS_NAMES_NUSCENES
    elif dataset == "semantic_kitti":
        CLASS_NAMES = CLASS_NAMES_SEMANTIC_KITTI

    log = (
        f"\nEpoch: {0:d} :\n"
        + " Loss = N/A"
        + f" - oAcc = {oAcc:.1f}"
        + f" - mAcc = {mAcc:.1f}"
        + f" - mIoU = {mIoU:.1f}"
    )
    print(log)
    # Per class score
    log = ""
    for i, s in enumerate(ious):
        log += f"{CLASS_NAMES[i]}: {100 * s:.1f} - "
    print(log[:-3])
    sys.stdout.flush()