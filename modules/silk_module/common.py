# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np

import torch

import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

from .silk_model import SiLKVGG as SiLK
from .silk_model import ParametricVGG
#from .silk_model import load_model_from_checkpoint

#from silk.silk.backbones.silk.silk import SiLKVGG as SiLK
#from silk.silk.backbones.superpoint.vgg import ParametricVGG
#from silk.silk.config.model import load_model_from_checkpoint

SILK_NMS = 0  # NMS radius, 0 = disabled
SILK_BORDER = 0  # remove detection on border, 0 = disabled
SILK_THRESHOLD = 1.0  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
SILK_TOP_K = 1000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptors",
    "probability",
)
SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
SILK_BACKBONE = ParametricVGG(
    use_max_pooling=False,
    padding=0,
    normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
)

def load_model_from_checkpoint(  # noqa: C901
    model: Union[pl.LightningModule, torch.nn.Module],
    checkpoint_path: str,
    strict: bool = True,
    device: Optional[str] = None,
    freeze: bool = False,
    eval: bool = False,
    map_name: Union[Dict[str, str], None] = None,
    remove_name: Union[List[str], None] = None,
    state_dict_key: Union[None, str] = "state_dict",
    state_dict_fn: Optional[Callable[[Any], Any]] = None,
):
    checkpoint = pl_load(checkpoint_path, device)

    if isinstance(model, pl.LightningModule):
        model.on_load_checkpoint(checkpoint)

    # get state dictionary
    if state_dict_key is not None:
        state_dict = checkpoint[state_dict_key]
    else:
        state_dict = checkpoint

    # remove names
    if remove_name is not None:
        for name in remove_name:
            del state_dict[name]

    # remap names
    if map_name is not None:
        for src, dst in map_name.items():
            if src not in state_dict:
                continue
            state_dict[dst] = state_dict[src]
            del state_dict[src]

    # apply custom changes to dict
    if state_dict_fn is not None:
        state_dict = state_dict_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model = model.to(device)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        eval = True

    if eval:
        model.eval()

    return model

def preprocess_image(image, device="cpu", res:int=4):
    _image = cv2.resize(image, (image.shape[1]//res, image.shape[0]//res), interpolation=cv2.INTER_AREA)
    _image = (_image / 255).astype(np.float32)
    _image = np.mean(_image, axis=2)
    
    images = [_image]
    images = np.stack(images)
    images = torch.tensor(images, device=device, dtype=torch.float32)
    images = images.unsqueeze(1)
    
    return images

def get_model(
    checkpoint,
    device,
    top_k=SILK_TOP_K,
    nms=SILK_NMS,
    default_outputs=SILK_DEFAULT_OUTPUT,
):
    # load model
    model = SiLK(
        in_channels=1,
        backbone=deepcopy(SILK_BACKBONE),
        detection_threshold=SILK_THRESHOLD,
        detection_top_k=top_k,
        nms_dist=nms,
        border_dist=SILK_BORDER,
        default_outputs=default_outputs,
        descriptor_scale_factor=SILK_SCALE_FACTOR,
        padding=0,
    )
    model = load_model_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
        device=device,
        freeze=True,
        eval=True,
    )
    return model