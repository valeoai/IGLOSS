python distill.py \
--dataset merged_datasets \
--path_dataset /datasets_local/ \
--log_path ./scratch/temp/ \
--config configs/pretrain/WI_768_droppath_gelu-nuScenes_mini-TIMM_DINOv2_ViTL-448x896-MLP_2048-65_epoch-checkpointing.yaml \
--multiprocessing-distributed