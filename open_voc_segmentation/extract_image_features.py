# Copyright 2024 - Valeo Comfort and Driving Assistance - valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import yaml
import torch
import argparse
import numpy as np
from PIL import Image
from models.image_teacher import ImageTeacher
from torchvision.transforms.functional import resize as vision_resize
import os
import torch
import numpy as np
from PIL import Image
import pickle
import cv2
from igloss_utils import (
    CLASS_NAMES_NUSCENES,
    CLASS_NAMES_SEMANTIC_KITTI
)
np.bool = np.bool_

def load_model_config(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_default_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config for pretraining"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Enable autocast for mix precision training",
    )
    parser.add_argument(
        "--im_path",
        type=str,
        required=True,        
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,        
    )
    
    return parser


if __name__ == "__main__":

    # Get args
    parser = get_default_parser()
    args = parser.parse_args()

    # Load config files
    config = load_model_config(args.config)

    # Load image model
    config["image_backbone"]["im_size"] = [448, 448]
    model_image = ImageTeacher(config)
    model_image = model_image.cuda()
    model_image = model_image.eval()
    
    if "semantic_kitti" in args.im_path:
        class_names = CLASS_NAMES_SEMANTIC_KITTI
    elif "nuscenes" in args.im_path: 
        class_names = CLASS_NAMES_NUSCENES
    else:
        raise ValueError(f"{dataset} is not supported.")
         
    root_templates = args.im_path
    cats = os.listdir(root_templates)
    feature_list = [None] * len(cats)

    cc = 1
    for ind, cat in enumerate(class_names):
        ref_images_sub = os.listdir(os.path.join(root_templates, f"{ind}_{cat}"))
        feature_list[ind] = []
        print(cat)

        for ref_image_n in ref_images_sub:
            print(cc)
            cc = cc + 1

            ref_image = np.array(Image.open(os.path.join(root_templates, f"{ind}_{cat}", ref_image_n)))
            ref_image = [ref_image / 255.0]
            ref_image = torch.tensor(np.array(ref_image, dtype=np.float32).transpose(0, 3, 1, 2))
            ref_image = vision_resize(ref_image, config["image_backbone"]["im_size"])

            # Extract features
            ref_image = ref_image.cuda(non_blocking=True)
            with torch.autocast("cuda", enabled=args.fp16):
                ref_image_features = model_image(ref_image)
            ref_image_features = torch.nn.functional.normalize(ref_image_features, p=2, dim=1) 
            ref_image_features_final = torch.mean(ref_image_features, dim=[2,3])
            feature_list[ind].append(ref_image_features_final)

    # save
    with open(args.save_path, "wb") as f:
        pickle.dump(feature_list, f)

 
 

 



