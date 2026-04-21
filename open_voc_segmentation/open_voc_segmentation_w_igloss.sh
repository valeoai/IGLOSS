
pip install transforms3d numpy==1.26.4

PROJECT_PATH="path/to/IGLOSS"

export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH

cd $PROJECT_PATH

CONFIG=${PROJECT_PATH}/configs/pretrain/WI_768_droppath_gelu-nuScenes_mini-TIMM_DINOv2_ViTL-448x896-MLP_2048-65_epoch-checkpointing.yaml
WI_CKPT=${PROJECT_PATH}/checkpoints/WI_768_droppath_gelu-dinov2_vitl-448x896-mlp_2048-65_epoch-checkpointing/ckpt_last.pth


DATASET="nuscenes" 
PATH_DATASET="/datasets_local/nuscenes/" 
TEMPLATE_PATH=${PROJECT_PATH}/open_voc_segmentation/templates/nuscenes_gpt_dinov2_features.pkl 

'''Use the following setting for Semantic KITTI dataset'''
# DATASET="semantic_kitti"
# PATH_DATASET="/datasets_local/semantic_kitti/" 
# TEMPLATE_PATH=${PROJECT_PATH}/open_voc_segmentation/templates/semantic_kitti_gpt_dinov2_features.pkl

python igloss_w_sclarplus.py \
--dataset ${DATASET} \
--path_dataset ${PATH_DATASET} \
--config ${CONFIG} \
--wi_pretrained_ckpt ${WI_CKPT} \
--fp16 \
--split val \
--templates ${TEMPLATE_PATH}


