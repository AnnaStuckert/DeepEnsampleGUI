import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from numpy import random
from torch.utils.data import (
    DataLoader,
    Dataset,
    SubsetRandomSampler,
    TensorDataset,
)
from torchvision.utils import save_image

from models import Unet

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.chdir(
    "/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/Unet_training"
)


def random_cutout(image, label, max_h=50, max_w=50):
    """
    Apply random cutout to both image and label.
    Args:
        image: The input image tensor.
        label: The label tensor.
        max_h: Maximum height of the cutout box.
        max_w: Maximum width of the cutout box.
    Returns:
        image: Image after cutout.
        label: Label after cutout.
    """
    _, h, w = image.shape
    cutout_height = random.randint(10, max_h)
    cutout_width = random.randint(10, max_w)

    # Randomly choose the position for the cutout
    top = random.randint(0, h - cutout_height)
    left = random.randint(0, w - cutout_width)

    # Apply the cutout to the image and label (set to 0)
    image[:, top : top + cutout_height, left : left + cutout_width] = 0
    label[:, top : top + cutout_height, left : left + cutout_width] = 0

    return image, label


def adjust_brightness(image, label, brightness_factor=0.2):
    """
    Adjust the brightness of the image and label.
    Args:
        image: The input image tensor.
        label: The label tensor.
        brightness_factor: Factor by which brightness is adjusted.
    Returns:
        image: Image after brightness adjustment.
        label: Label after brightness adjustment.
    """
    image = TF.adjust_brightness(
        image, 1 + (random.random() * 2 - 1) * brightness_factor
    )
    # Note: Brightness doesn't affect the label, so we leave it unchanged
    return image, label


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def random_jitter(image, max_jitter=0.1):
    """
    Apply random jittering to an image by adding noise to its pixel values.
    Args:
        image: The input image tensor.
        max_jitter: The maximum amount of jitter to apply to each pixel.
    Returns:
        image: The image with jitter applied.
    """
    # Generate random noise with a normal distribution
    noise = torch.randn_like(image) * max_jitter  # Gaussian noise
    image = image + noise

    # Clip the values to be in the valid range [0, 1] for images
    image = torch.clamp(image, 0, 1)

    return image


def motion_blur(image, kernel_size=5, angle=45):
    """
    Apply motion blur to an image using a convolution with a motion blur kernel.
    Args:
        image: The input image tensor.
        kernel_size: The size of the blur kernel.
        angle: The angle of the motion.
    Returns:
        image: The image after motion blur.
    """
    # Create motion blur kernel
    kernel = torch.zeros((kernel_size, kernel_size))

    # Define the direction of the blur (this could be any angle, here we use horizontal motion)
    center = kernel_size // 2
    angle_rad = torch.tensor(angle * torch.pi / 180)  # Convert to radians

    # Apply a simple horizontal motion blur
    for i in range(kernel_size):
        kernel[center, i] = 1

    # Normalize the kernel
    kernel = kernel / kernel.sum()

    # Reshape kernel for convolution (batch size, channels, kernel size)
    kernel = kernel.unsqueeze(0).unsqueeze(
        0
    )  # Shape (1, 1, kernel_size, kernel_size)

    # Apply the kernel using convolution
    blurred_image = F.conv2d(
        image.unsqueeze(0), kernel, padding=kernel_size // 2
    )
    return blurred_image.squeeze(0)


class facemapdataset(Dataset):
    # def __init__(self, data_file="data/dolensek_facemap_softlabels_224.pt",
    # def __init__(self, data_file="data\dolensek_facemap_softlabels_224_TEST_DIF_KP.pt",
    def __init__(
        self,
        data_file="/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/Unet_training/data/dataset.pt",
        transform=None,
        rotation_degrees=(
            15,
            30,
        ),  # Rotation angle range from 15 to 30 degrees
        zoom_range=(
            0.8,
            1.5,
        ),  # Zoom range from 0.8 (zoom out) to 1.5 (zoom in)
        blur_radius=(1, 2),  # Tuple for Gaussian blur radius range
        cutout_prob=0.2,  # Probability of applying cutout
        brightness_prob=0.2,  # Probability of applying brightness adjustment
        brightness_factor=0.5,  # Max factor for brightness adjustment
        motion_blur_prob=0.2,  # Probability of applying motion blur
        motion_blur_kernel_size=5,  # Size of the motion blur kernel
        motion_blur_angle=45,  # Angle of the motion blur
        jitter_prob=0.2,  # Probability of applying random jitter
        jitter_max=0.1,
    ):  # Maximum jitter value (standard deviation)
        super().__init__()
        self.transform = transform
        self.rotation_degrees = rotation_degrees
        self.zoom_range = zoom_range
        self.blur_radius = blur_radius
        self.cutout_prob = cutout_prob
        self.brightness_prob = brightness_prob
        self.brightness_factor = brightness_factor
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel_size = motion_blur_kernel_size
        self.motion_blur_angle = motion_blur_angle
        self.jitter_prob = jitter_prob
        self.jitter_max = jitter_max
        # self.data, _, self.targets = torch.load(data_file)
        self.data, self.targets = torch.load(data_file)

    # def __len__(self):
    #    return len(self.data) * 5  # Return length * 5 for augmented versions
    def __len__(self):
        return len(self.data) * 10  # Return length * 10 for augmented versions

    def __getitem__(self, index):
        # Ensure the index stays within bounds by using modulo with the original dataset size
        base_index = index % len(
            self.data
        )  # This will prevent out-of-bounds errors
        aug_type = index // len(
            self.data
        )  # This will determine which augmentation to apply

        # Load the original image and label
        image, label = (
            self.data[base_index].clone(),
            self.targets[base_index].clone(),
        )

        # Apply the augmentation based on the `aug_type`
        if self.transform is not None:
            if aug_type == 1:  # Flipping
                image = image.flip([2])
                label = label.flip([2])
            elif aug_type == 2:  # Rotation
                angle = random.uniform(
                    -self.rotation_degrees[1], self.rotation_degrees[1]
                )
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            elif aug_type == 3:  # Zooming
                scale_factor = random.uniform(
                    self.zoom_range[0], self.zoom_range[1]
                )
                image = self.zoom(image, scale_factor)
                label = self.zoom(label, scale_factor)
            elif aug_type == 4:  # Gaussian Blur
                radius = (
                    torch.rand(1).item()
                    * (self.blur_radius[1] - self.blur_radius[0])
                    + self.blur_radius[0]
                )
                image = TF.gaussian_blur(image, kernel_size=int(radius))
                # Do not apply blur to the label

            # Apply random cutout with probability
            if random.random() < self.cutout_prob:
                image, label = random_cutout(image, label)

            # Apply random brightness adjustment with probability
            if random.random() < self.brightness_prob:
                image, _ = adjust_brightness(
                    image, label, self.brightness_factor
                )
                # Note that the label is not being adjusted, only the image

            # Apply motion blur with probability
            if random.random() < self.motion_blur_prob:
                image = motion_blur(
                    image, self.motion_blur_kernel_size, self.motion_blur_angle
                )

            # Apply random jittering with probability
            if random.random() < self.jitter_prob:
                image = random_jitter(image, self.jitter_max)

        return image, label

    def zoom(self, img, scale_factor):
        # Calculate new dimensions
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize and center-crop back to the original size
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        return img


# class facemapdataset(Dataset):
#     def __init__(self, data_file="data/dataset.pt", transform=None):
#         super().__init__()

#         self.transform = transform
#         # self.data, _, self.targets = torch.load(data_file)
#         self.data, self.targets = torch.load(data_file)

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, index):
#         image, label = self.data[index].clone(), self.targets[index].clone()
#         if (self.transform is not None) and (torch.rand(1) > 0.5):
#             image = image.flip([2])
#             label = label.flip([2])
#         return image, label


### Make dataset
dataset = facemapdataset(transform="flip")

x = dataset[0][0]
dim = x.shape[-1]
print("Using %d size of images" % dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * N), int(0.8 * N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * N), N))
batch_size = 4
# Initialize loss and metrics
loss_fun = torch.nn.MSELoss(reduction="sum")

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print(
    "Num. train = %d, Num. val = %d, Num. test = %d"
    % (num_train, num_valid, num_test)
)

# Initialize dataloaders
loader_train = DataLoader(
    dataset=dataset,
    drop_last=False,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=train_sampler,
)
loader_valid = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=valid_sampler,
)
loader_test = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=1,
    pin_memory=True,
    sampler=test_sampler,
)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

### hyperparam
lr = 5e-4
num_epochs = 5

# num_input_channels = 1  # Change this to the desired number of input channels
# num_output_classes = 24  # Change this to the desired number of output classes


model = Unet()
# timm.create_model('vit_base_patch8_224',
#        pretrained=True,in_chans=1,num_classes=num_output_classes)

model = model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%2f M" % (nParam / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
minLoss = 1e6
convIter = 0
patience = 5
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    tr_loss = 0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = model(inputs)

        loss = loss_fun((scores), ((labels)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, nTrain, loss.item()
            )
        )
        tr_loss += loss.item()
    train_loss.append(tr_loss / (i + 1))

    with torch.no_grad():
        val_loss = 0
        for i, (inputs, labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores, fmap = model(inputs)
            loss = loss_fun((scores), ((labels)))
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

        valid_loss.append(val_loss)

        print("Val. loss :%.4f" % val_loss)

        labels = labels.squeeze().detach().cpu().numpy()
        scores = scores.squeeze().detach().cpu().numpy()
        img = inputs.squeeze().detach().cpu().numpy()
        fmap = inputs.mean(1).squeeze().detach().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(16, 12))
        for i in range(batch_size):
            plt.subplot(batch_size, 3, 3 * i + 1)
            plt.imshow(labels[i])
            plt.subplot(batch_size, 3, 3 * i + 2)
            plt.imshow(scores[i] * img[i])
            plt.subplot(batch_size, 3, 3 * i + 3)
            plt.imshow(fmap[i])

        plt.tight_layout()

        plt.savefig("logs/epoch_%03d.jpg" % epoch)

        if minLoss > val_loss:
            convEpoch = epoch
            minLoss = val_loss
            convIter = 0
            # torch.save(model.state_dict(),'models/best_model.pt')
        else:
            convIter += 1

        if convIter == patience:
            print(
                "Converged at epoch %d with val. loss %.4f"
                % (convEpoch + 1, minLoss)
            )
            break
plt.clf()
plt.plot(train_loss, label="Training")
plt.plot(valid_loss, label="Valid")
plt.plot(convEpoch, valid_loss[convEpoch], "x", label="Final Model")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.pdf")

### Load best model for inference
with torch.no_grad():
    val_loss = 0

    for i, (inputs, labels) in enumerate(loader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, fmap = model(inputs)
        loss = loss_fun((scores), ((labels)))
        val_loss += loss.item()

        img = inputs.squeeze().detach().cpu().numpy()
        pred = scores.squeeze().detach().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        fmap = fmap.mean(1).squeeze().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(12, 4))
        plt.subplot(141)
        plt.imshow(img, cmap="gray")
        plt.subplot(142)
        plt.imshow(labels)
        plt.subplot(143)
        plt.imshow(pred)
        plt.subplot(144)
        plt.imshow(fmap)

        plt.tight_layout()
        plt.savefig("preds/test_%03d.jpg" % i)

    val_loss = val_loss / (i + 1)

    print("Test. loss :%.4f" % val_loss)
