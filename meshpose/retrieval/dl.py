import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm
from loguru import logger

from typing import List

from .base import BaseImageRetrieval
from meshpose.data import ViewDescription, QueryView
from meshpose.utils import tqdm_description

TORCH_AVAILABLE = True
try:
    import torch
    from torch import nn
    
    import torchvision
    from torchvision import models
except:
    TORCH_AVAILABLE = False

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
    def __init__(self, device:str="cuda", n:float=0.5):
        assert TORCH_AVAILABLE, logger.error("Torch and torchvision are not available. Please install them to use this feature extractor.")
        
        # Init params
        self.n = n
        self.device = device
        
        # Create model
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        encoder = models.efficientnet_v2_s(weights=weights)
        
        self.encoder = Encoder(encoder)
        self.encoder.to(device)
        self.encoder.eval()
        
        # Create im transform
        self.transform = weights.transforms()
        
        # Reduce size
        self.pool = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        
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
            
    def query(self, query_desc:ViewDescription)-> List[int]:
        assert self.descriptions is not None, logger.error("You must train the model before querying.")
        # Get image
        image = cv2.cvtColor(query_desc.view.image, cv2.COLOR_BGR2RGB)
        
        # Inference
        desc = self._inference(image)
        desc = torch.unsqueeze(desc, 0)
        
        # Find distances
        dist = torch.cdist(desc, self.descriptions, p=2)[0]
        
        # Take top N
        idx = torch.argsort(dist)
        n_ret = int(len(idx) * self.n)
        idx = idx[:n_ret].cpu().numpy()
        return idx