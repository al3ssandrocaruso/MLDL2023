import torch
import torch.nn as nn
import torch.optim as optim

import utils.loaders
from utils.loaders import EpicKitchensDataset

PATH = "rgb_imagenet.pt"

model = torch.load(PATH)  # load the model from the pretrained i3d
model.eval()  # set the model to evaluation mode

dataset = utils.loaders.EpicKitchensDataset('RGB','test',)

with torch.no_grad():
    output = model(input_batch)


