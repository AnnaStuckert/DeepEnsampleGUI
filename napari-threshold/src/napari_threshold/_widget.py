"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from napari.types import ImageData, LabelsData
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_li, threshold_multiotsu
from collections import Counter
from .models import UNet

import os
import numpy as np
import torch
import torch.nn as nn

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
    
@magic_factory(call_button="Run", filter_selected={"choices": ['otsu', 'li', 'multi_otsu', 'U-net']})
def threshold_f(selected_image: ImageData, filter_selected='U-net') -> LabelsData:
    # Check if the image is RGB (3D) or already grayscale (2D)
    if selected_image.ndim == 3 and selected_image.shape[-1] == 3:
        gray_ = rgb2gray(selected_image)  # Convert to grayscale if RGB
    else:
        gray_ = selected_image  # Use the image directly if already grayscale

    if filter_selected == 'otsu':
        ths = threshold_otsu(gray_)  # Determine threshold with Otsu method
        mask = gray_ > ths
    elif filter_selected == 'li':
        from skimage.filters import threshold_li
        ths = threshold_li(gray_)  # Determine threshold with Li's method
        mask = gray_ > ths
    elif filter_selected == 'multi_otsu':
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(gray_)
        mask = np.digitize(gray_, bins=thresholds)
    elif filter_selected == 'U-net':
        import torch
        import torchvision.transforms as T
        from PIL import Image

        # Load the pre-trained U-net model
        model_path = "models\\cell_segmentation_unet.pth"
        model = UNet()  # Replace `UNet` with the correct model class used during training
    
        # Load the state dictionary
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode

        # Preprocess the image for the U-net model
        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((512, 512)),  
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Match training normalization
        ])

        input_tensor = preprocess(gray_).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Post-process the output to create a binary mask
        #upscaled_prediction = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        output_resized = torch.nn.functional.interpolate(output, size=gray_.shape, mode='bilinear', align_corners=False)

        mask = output_resized.squeeze().numpy() > 0.5  # Apply a threshold to the U-net output

    return mask  

@magic_factory(result_widget=True)
def leaf_area(mask: "napari.layers.Labels"):
    current_mask = mask.data
    dico = dict(Counter(current_mask.flatten())) # Total number of black and white pixel
    labels_leaf_area = dico[True]
    print('Leaf Area:',labels_leaf_area) # Get total number of white pixel
    return labels_leaf_area

# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")