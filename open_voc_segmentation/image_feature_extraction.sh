
pip install transforms3d numpy==1.26.4

PROJECT_PATH="path/to/IGLOSS"
cd $PROJECT_PATH

export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH

CONFIG=${PROJECT_PATH}/configs/pretrain/WI_768_droppath_gelu-nuScenes_mini-TIMM_DINOv2_ViTL-448x896-MLP_2048-65_epoch-checkpointing.yaml

TEMPLATE_IMAGES_PATH=${PROJECT_PATH}/open_voc_segmentation/templates/images/nuscenes_gpt_templates 
SAVE_PATH=${PROJECT_PATH}/open_voc_segmentation/templates/nuscenes_gpt_dinov2_features.pkl 

'''Use the following setting for Semantic KITTI dataset'''
# TEMPLATE_IMAGES_PATH=${PROJECT_PATH}/open_voc_segmentation/templates/images/semantic_kitti_gpt_templates
# SAVE_PATH=${PROJECT_PATH}/open_voc_segmentation/templates/semantic_kitti_gpt_dinov2_features.pkl

python extract_image_features_with_timm.py \
--config ${CONFIG} \
--fp16 \
--im_path ${TEMPLATE_IMAGES_PATH} \ 
--save_path ${SAVE_PATH}


