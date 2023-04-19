import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import utils.loaders
from utils.loaders import EpicKitchensDataset

PATH1 = "rgb_imagenet.pt"
PATH2 = "C:\\tmp\\MLDLproject\\dataTestPoint2"
batch_size = 1024


model = torch.load(PATH1)  # load the model from the pretrained i3d
model.eval()  # set the model to evaluation mode

dataset_conf = utils.loaders.DatasetConf(PATH2, 1)

data_in = utils.loaders.EpicKitchensDataset(split='D1',
                                            modalities=['RGB'],
                                            mode='test',
                                            dataset_conf=dataset_conf,
                                            num_frames_per_clip=10,
                                            num_clips=5,
                                            dense_sampling=True)
data_loader = torch.utils.data.DataLoader(data_in, batch_size, shuffle=True)

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        outputs = model(inputs)
