"""/segmentation_with_channel_regularization/utils/visualization.py

Helper utiliities for visualizing images and segmentations.
"""

import numpy as np
from PIL import Image


def black_to_transparency(img):
    """Convert black pixels to alpha."""
    x = np.asarray(img.convert('RGBA')).copy()
    x[:, :, 3] = (255 * (x[:, :, :3] != 0).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)


def overlay_segmentation(image, segmentation_image, blend_alpha=0.3):
    """Overlay the segmentation on the original image

    Black pixels are converted to alpha.
    """
    return Image.blend(image,
                       black_to_transparency(segmentation_image).convert('RGB'),
                       alpha=blend_alpha)

