import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
from loguru import logger

from typing import List

from .base import BaseImageRetrieval
from mesh_pose.data import ViewDescription, PresetView
from mesh_pose.utils import tqdm_description

TORCH_AVAILABLE = True
try:
    import torch
    from torch import nn
    
    import torchvision
    from torchvision import models
except:
    TORCH_AVAILABLE = False

BACKBONE_DICT = {
    "densenet169": (
        models.densenet169,
        models.DenseNet169_Weights.DEFAULT
        ),
    "densenet201": (
        models.densenet201,
        models.DenseNet201_Weights.DEFAULT
        ),
    "efficientnet-v2-s": (
        models.efficientnet_v2_s,
        models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        ),
    "efficientnet-v2-m": (
        models.efficientnet_v2_m,
        models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        )
}

class Encoder(nn.Module):
    def __init__(self, encoder_model):
        super(Encoder, self).__init__()
        self.original_model = encoder_model
        
    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            x = v(features[-1])
            features.append(x)
        return features[-2]

class DLRetrieval(BaseImageRetrieval):
    def __init__(self, backbone_type="efficientnet-v2-s", device:str="cuda", n:float=0.5):
        assert TORCH_AVAILABLE, logger.error("Torch and torchvision are not available. Please install them to use this feature extractor.")
        assert backbone_type in BACKBONE_DICT.keys(), logger.error(f"Backbone type {backbone_type} is not supported.")
        
        # Init params
        self.n = n
        self.bbt = backbone_type
        self.device = device
        
        # Create model
        model, weights = BACKBONE_DICT[backbone_type]
        encoder = model(weights=weights)
        
        self.encoder = Encoder(encoder)
        self.encoder.to(device)
        self.encoder.eval()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=3)
        
        # Create im transform
        self.transform = weights.transforms()
        
        self.descriptions = None
    
    def _inference(self, image:np.ndarray):
        # Preprocess
        image_pil = Image.fromarray(image)
        image_pt = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            features = self.encoder(image_pt)
            desc = self.pool(features) # [1, 256, 12, 12] -> [1, 256, 4, 4]
            desc = torch.flatten(desc) # [1, 256, 4, 4] -> [1, 4096]
        return desc
    
    def train(self, views_desc:List[ViewDescription]):    
        self.descriptions = []
        
        for vd in tqdm(views_desc, desc=tqdm_description("mesh_pose.retrieval.dl", "Image encoding")):
            image = cv2.cvtColor(vd.view.image, cv2.COLOR_BGR2RGB)
            desc = self._inference(image)
            self.descriptions.append(desc)
        
        self.descriptions = torch.stack(self.descriptions)
        print("yo")
            
    def query(self, view:PresetView)-> List[int]:
        assert self.descriptions is not None, logger.error("You must train the model before querying.")
        # Get image
        image = cv2.cvtColor(view.image, cv2.COLOR_BGR2RGB)
        
        # Inference
        desc = self._inference(image)
        desc = torch.unsqueeze(desc, 0)
        
        # Find distances
        dist = torch.cdist(desc, self.descriptions, p=2)[0]
        
        # Take top N
        idx = torch.argsort(dist)
        n_ret = int(len(self.descriptions) * self.n)
        idx = idx[:n_ret].cpu().numpy()
        return idx