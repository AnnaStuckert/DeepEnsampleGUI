"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import os
import pdb
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from magicgui import magic_factory
from napari.types import ImageData, LabelsData
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.color import rgb2gray
from skimage.filters import threshold_li, threshold_multiotsu, threshold_otsu
from skimage.io import imread
from torch.nn.functional import sigmoid, softmax, softplus

from .models import Unet
from .models3 import UNet2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


@magic_factory(
    call_button="Run",
    filter_selected={
        "choices": ["otsu", "li", "multi_otsu", "U-net", "Unet_mouse"]
    },
)
def threshold_f(
    selected_image: ImageData, filter_selected="Unet_mouse"
) -> LabelsData:
    # Check if the image is RGB (3D) or already grayscale (2D)
    if selected_image.ndim == 3 and selected_image.shape[-1] == 3:
        gray_ = rgb2gray(selected_image)  # Convert to grayscale if RGB
    else:
        gray_ = selected_image  # Use the image directly if already grayscale

    if filter_selected == "otsu":
        ths = threshold_otsu(gray_)  # Determine threshold with Otsu method
        mask = gray_ > ths
    elif filter_selected == "li":
        from skimage.filters import threshold_li

        ths = threshold_li(gray_)  # Determine threshold with Li's method
        mask = gray_ > ths
    elif filter_selected == "multi_otsu":
        from skimage.filters import threshold_multiotsu

        thresholds = threshold_multiotsu(gray_)
        mask = np.digitize(gray_, bins=thresholds)
    elif filter_selected == "U-net":
        import torch
        import torchvision.transforms as T
        from PIL import Image

        # Load the pre-trained U-net model
        model_path = "/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/src/napari_threshold/cell_segmentation_unet.pth"
        model = (
            UNet2()
        )  # Replace `UNet` with the correct model class used during training

        # Load the state dictionary
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode

        # Preprocess the image for the U-net model
        preprocess = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.5], std=[0.5]
                ),  # Match training normalization
            ]
        )

        input_tensor = preprocess(gray_).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Post-process the output to create a binary mask
        output_resized = torch.nn.functional.interpolate(
            output, size=gray_.shape, mode="bilinear", align_corners=False
        )

        mask = (
            output_resized.squeeze().numpy() > 0.5
        )  # Apply a threshold to the U-net output
    elif filter_selected == "Unet_mouse":
        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        # Define the path to your single image
        # img_path = "/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/src/napari_threshold/frame_0002.jpg"  # Replace with your image path
        # Load the single image
        # img = Image.open(img_path).convert('L')  # Convert to grayscale ('L')
        # Define any transformations (optional, for example, converting to tensor)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor
            ]
        )

        # Load the single image

        import numpy as np
        from PIL import Image

        # Assuming gray_ is a NumPy array that you want to convert to a PIL image
        if gray_.ndim == 2:  # Grayscale image (2D)
            img = Image.fromarray(
                np.uint8(gray_ * 255)
            )  # If gray_ is in the range [0, 1], scale to [0, 255]
        else:  # RGB image (3D)
            img = Image.fromarray(np.uint8(gray_))

        # Resize the image
        resize_factor = 0.2
        min_size = 32
        new_width = max(int(img.width * resize_factor), min_size)
        new_height = max(int(img.height * resize_factor), min_size)

        # Ensure the resized dimensions are even
        new_width += new_width % 2  # Make width even
        new_height += new_height % 2  # Make height even

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # img = Image.open(gray_).convert("L")  # Convert to grayscale ('L')
        # img = Image.open(gray_)
        # Resize the image
        # resize_factor = 0.2
        # min_size = 32
        # new_width = max(int(img.width * resize_factor), min_size)
        # new_height = max(int(img.height * resize_factor), min_size)

        # # Ensure the resized dimensions are even
        # new_width += new_width % 2  # Make width even
        # new_height += new_height % 2  # Make height even

        # img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Apply the transformation (e.g., converting to tensor)
        img_tensor = transform(img)

        # Define the save path for the .pt file
        save_path = "/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/src/napari_threshold/single_image.pt"  # Change to the path where you want to save the file

        # Save the image tensor to the .pt file
        torch.save(img_tensor, save_path)
        print(f"Image saved to {save_path}")

        import random

        import torch
        import torchvision.transforms.functional as TF
        from torch.utils.data import Dataset

        class facemapdataset(Dataset):
            def __init__(
                self,
                data_file="/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/src/napari_threshold/single_image.pt",
                transform=None,
            ):
                super().__init__()
                self.transform = transform
                # Load the data (single image tensor)
                self.data = torch.load(data_file)

            def __len__(self):
                return len(
                    self.data
                )  # We are using a single image dataset, no augmentation.

            def __getitem__(self, index):
                # Load the image and label
                image = self.data[index].clone()

                # No augmentation or transformation is applied to the image or label
                return image

        # Initialize the dataset
        dataset = facemapdataset()

        # Example of retrieving a sample from the dataset
        image = dataset[0]
        print(f"Image shape: {image.shape}")

        import matplotlib.pyplot as plt
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from .models import Unet

        # Initialize the Unet model
        model = Unet()

        # Load the model weights
        model.load_state_dict(
            torch.load(
                "/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/src/napari_threshold/model_weights.pth",
                map_location=device,
            )
        )
        model.eval()  # Set the model to evaluation mode

        # Assuming image is already loaded
        image = dataset[0]  # Shape: [244, 388]
        image = image.unsqueeze(0).unsqueeze(
            0
        )  # Now the shape is [1, 1, 244, 388]

        # Move the image to the device (GPU or CPU)
        inputs = image.to(device)

        # Perform inference with the model
        with torch.no_grad():
            scores, fmap = model(inputs)

            # Optionally, process the results (e.g., loss, visualization)
            img = inputs.squeeze().detach().cpu().numpy()
            pred = scores.squeeze().detach().cpu().numpy()
            fmap = fmap.mean(1).squeeze().cpu().numpy()

        # # Post-process the output to create a binary mask
        # output_resized = torch.nn.functional.interpolate(
        #     output, size=gray_.shape, mode="bilinear", align_corners=False
        # )

        # mask = pred
        # mask = pred  # Assuming the model outputs logits and you're using 0.5 as the threshold
        # mask = mask.astype(
        #    np.uint8
        # )  # Convert the mask to a binary mask (0 or 1)

        # Post-process the output to create a binary mask
        # Interpolation for resizing the output to match the original gray image size
        output_resized = torch.nn.functional.interpolate(
            scores, size=gray_.shape, mode="bilinear", align_corners=False
        )

        # Apply threshold to the prediction (if needed)
        mask = (
            output_resized.squeeze().detach().cpu().numpy() > 0.002
        )  # Binary mask
        mask = mask.astype(np.uint8)  # Convert to uint8 for visualization

        # Display the results
        # plt.imshow(mask, cmap="gray")
        # plt.title("Predicted Mask")
        # plt.show()
        plt.subplot(143)
        plt.imshow(img, cmap="gray")
        plt.title("image")
        plt.show()
        plt.subplot(144)
        plt.imshow(pred)
        plt.title("pred")
        plt.colorbar()
        plt.show()
        # Apply a threshold to the U-net output

    return mask


# @magic_factory(
#     call_button="Run",
#     filter_selected={
#         "choices": ["otsu", "li", "multi_otsu", "U-net", "Unet_mouse"]
#     },
# )
# def threshold_f(
#     selected_image: ImageData, filter_selected="Unet_mouse"
# ) -> LabelsData:
#     # Check if the image is RGB (3D) or already grayscale (2D)
#     if selected_image.ndim == 3 and selected_image.shape[-1] == 3:
#         gray_ = rgb2gray(selected_image)  # Convert to grayscale if RGB
#     else:
#         gray_ = selected_image  # Use the image directly if already grayscale

#     if filter_selected == "otsu":
#         ths = threshold_otsu(gray_)  # Determine threshold with Otsu method
#         mask = gray_ > ths
#     elif filter_selected == "li":
#         from skimage.filters import threshold_li

#         ths = threshold_li(gray_)  # Determine threshold with Li's method
#         mask = gray_ > ths
#     elif filter_selected == "multi_otsu":
#         from skimage.filters import threshold_multiotsu

#         thresholds = threshold_multiotsu(gray_)
#         mask = np.digitize(gray_, bins=thresholds)
#     elif filter_selected == "U-net":
#         import torch
#         import torchvision.transforms as T
#         from PIL import Image

#         # Load the pre-trained U-net model
#         model_path = "/Users/annastuckert/Documents/GitHub/DeepEnsampleGUI/napari-threshold/src/napari_threshold/cell_segmentation_unet.pth"
#         model = (
#             UNet2()
#         )  # Replace `UNet` with the correct model class used during training

#         # Load the state dictionary
#         state_dict = torch.load(model_path, weights_only=True)
#         model.load_state_dict(state_dict)
#         model.eval()  # Set the model to evaluation mode

#         # Preprocess the image for the U-net model
#         preprocess = T.Compose(
#             [
#                 T.ToPILImage(),
#                 T.Resize((512, 512)),
#                 T.ToTensor(),
#                 T.Normalize(
#                     mean=[0.5], std=[0.5]
#                 ),  # Match training normalization
#             ]
#         )

#         input_tensor = preprocess(gray_).unsqueeze(0)  # Add batch dimension

#         # Perform inference
#         with torch.no_grad():
#             output = model(input_tensor)

#         # Post-process the output to create a binary mask
#         # upscaled_prediction = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
#         output_resized = torch.nn.functional.interpolate(
#             output, size=gray_.shape, mode="bilinear", align_corners=False
#         )

#         mask = (
#             output_resized.squeeze().numpy() > 0.5
#         )  # Apply a threshold to the U-net output
#     elif filter_selected == "Unet_mouse":
#             ## INSERT CODE HERE

#     return mask


@magic_factory(result_widget=True)
def leaf_area(mask: "napari.layers.Labels"):
    current_mask = mask.data
    dico = dict(
        Counter(current_mask.flatten())
    )  # Total number of black and white pixel
    labels_leaf_area = dico[True]
    print("Leaf Area:", labels_leaf_area)  # Get total number of white pixel
    return labels_leaf_area


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
