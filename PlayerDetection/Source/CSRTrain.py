import h5py
from scipy import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from scipy import io
import torchvision.transforms.functional as F
from CSRNet import CSRNet
import torch
from torchvision import transforms as t

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

transform = t.Compose([
    t.ToTensor(), t.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

model = CSRNet().to(device)
model.train()

checkpoint = torch.load("PlayerDetection/Res/Weights_PreTrained/CSRNet_Weights.pth", map_location="cpu")
model.load_state_dict(checkpoint)

img_path = "PlayerDetection\Res\Images\Test_IMG.JPG"

print("Original Image")
plt.imshow(plt.imread(img_path))
plt.show()

img = transform(Image.open(img_path).convert("RGB")).to(device)
output = model(img.unsqueeze(0))
print("Predicted Count: ", int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
plt.imshow(temp, cmap=cm.jet)
plt.show()