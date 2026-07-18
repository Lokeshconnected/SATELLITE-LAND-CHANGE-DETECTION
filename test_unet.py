import torch
from unet_model import UNet

# create model
model = UNet()

# fake input image (batch=1, channels=6, size=256x256)
x = torch.randn(1,6,256,256)

# run model
y = model(x)

print("Output shape:", y.shape)