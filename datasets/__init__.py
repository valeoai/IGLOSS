from .im_pc_dataset import CollateDistillation
from .merged_datasets import MergedDatasetsDistill
from .nuscenes_for_scalr import (
    NuScenesDistill,
    NuScenesMiniDistill,
    NuScenesMiniSemSeg,
    NuScenesSemSeg,
)
from .pandaset_for_scalr import (
    PandaSet64Distill,
    Pandaset64SemSeg,
    PandaSetGTDistill,
    PandasetGTSemSeg,
)
from .pc_dataset import Collate
from .semantic_kitti_for_scalr import (
    SemanticKITTIDistill,
    SemanticKITTISemSeg,
)

__all__ = {Collate, CollateDistillation}

LIST_DATASETS = {
    "nuscenes": NuScenesSemSeg,
    "nuscenes_mini": NuScenesMiniSemSeg,
    "semantic_kitti": SemanticKITTISemSeg,
    "panda64": Pandaset64SemSeg,
    "pandagt": PandasetGTSemSeg,
}

LIST_DATASETS_DISTILL = {
    "nuscenes": NuScenesDistill,
    "nuscenes_mini": NuScenesMiniDistill,
    "semantic_kitti": SemanticKITTIDistill,
    "panda64": PandaSet64Distill,
    "pandagt": PandaSetGTDistill,
    "merged_datasets": MergedDatasetsDistill,
}
