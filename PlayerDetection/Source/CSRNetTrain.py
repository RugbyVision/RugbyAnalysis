import h5py
from scipy import io
from PIL import Image
import numpy as np
from matplotlib import pyplot, cm as plt, cm
from scipy.ndimage import gaussian_filter
import scipy
import torchvision.transforms.functional as F
from CSRNetwork import CSRNet
import torch
from torchvision import transforms as t

transform = t.Compose([
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

model = CSRNet()

checkpoint = torch.load("PlayerDetection/Resource/Weights/weights.pth", map_location="cpu")
model.load_state_dict(checkpoint)

img_path = ""