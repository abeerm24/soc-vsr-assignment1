# Train the model here
from operator import mod
import torch.nn as nn
import torch
from model import Net 
import numpy as np
import torch.nn.functional as F
import glob
import PIL.Image as pil_image

model = Net()
FILE = "nn-model-8x8-16b.pth"
model.load_state_dict(torch.load(FILE))
model.eval()

#Define hyper-parameters
num_epochs = 5
learning_rate = 1.5e-3
patch_size = 8
stride = 6

#Initialize loss function and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Location of lr images
lrImages_dir = "training-imgs/T91-lr"

#Train the model
for epoch in range(num_epochs):
    fileNum = 0
    for image_path in sorted(glob.glob('{}/*'.format(lrImages_dir))):

        #Open the hr and lr images
        hrimage_path = "training-imgs/T91-hr/" + image_path[21:]
        hr = np.array(pil_image.open(hrimage_path).convert("RGB"))
        lr = np.array(pil_image.open(image_path).convert("RGB"))
        
        #Create 4x4 patches from  the image
        lr_patches = []
        hr_patches = []
        for i in range(0, lr.shape[0] - patch_size + 1, stride):
            for j in range(0, lr.shape[1] - patch_size + 1, stride):
                lr_patches.append(lr[i:i + patch_size, j:j + patch_size])
                hr_patches.append(hr[i:i + patch_size, j:j + patch_size])
        lr_patches = torch.tensor(np.array(lr_patches),dtype=torch.float)
        hr_patches = torch.tensor(np.array(hr_patches),dtype=torch.float)

        if(fileNum==0 and epoch==0): print("Starting training")
        output_patches = model(lr_patches)

        if(fileNum==0 and epoch==0): print("Model works")
        (N,n,m,c) = hr_patches.shape
        output_patches = output_patches.view(N,n,m,c)

        cost = loss(output_patches,hr_patches)
        if(fileNum==0 and epoch==0): print("Loss calculation successful")

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (fileNum+1)%10==0:
            statement = "Epoch: " + str(epoch) + ", filenum: " + str(fileNum+1) + ", loss: " + str(cost.item())
            print(statement)
        fileNum+=1

FILE = "nn-model-8x8-16b.pth"
torch.save(model.state_dict(),FILE)
