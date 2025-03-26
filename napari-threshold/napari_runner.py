import os

import napari
from skimage import io


def add_label_layer(viewer, layer):
    """Automatically add a label layer with the same name as the image."""
    if isinstance(layer, napari.layers.Image):
        # Extract the filename without the extension to name the label layer
        image_name = os.path.splitext(os.path.basename(layer.source))[0]

        # Add a label layer with the same name as the image
        viewer.add_labels(layer.data, name=f"{image_name}_labels")


# Create a Napari viewer
viewer = napari.Viewer()

# Connect the function to the viewer's layer added event
viewer.layers.events.inserted.connect(
    lambda event: add_label_layer(viewer, event.layer)
)

# Start the Napari viewer
napari.run()
