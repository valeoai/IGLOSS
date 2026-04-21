python finetune.py \
--dataset nuscenes \
--path_dataset /datasets_local/nuscenes/ \
--log_path ~/scratch/ \
--config_pretrain configs/pretrain/WI_768_droppath_gelu-nuScenes_mini-TIMM_DINOv2_ViTL-448x896-MLP_2048-65_epoch-checkpointing.yaml \
--config_downstream configs/downstream/nuscenes/WI_linprob.yaml \
--pretrained_ckpt checkpoints/WI_768_droppath_gelu-dinov2_vitl-448x896-mlp_2048-65_epoch-checkpointing/ckpt_last.pth \
--multiprocessing-distributed \
--linprob
