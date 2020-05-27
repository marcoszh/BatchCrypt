import tensorflow as tf
import numpy as np

def flip(x):
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)

    return x


def color(x):
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def zoom(x):
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    scale = np.random.uniform(0.8, 1.0)

    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    box = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=[box], box_indices=[0], crop_size=(32, 32))
        # Return a random crop
        return crops[0]

    x = random_crop(x)

    return x


def augment_img(x, y):
    if np.random.uniform(0.0, 1.0) > 0.5:
        x = flip(x)
    if np.random.uniform(0.0, 1.0) > 0.5:
        x = color(x)
    if np.random.uniform(0.0, 1.0) > 0.75:
        x = zoom(x)
    return x, y
