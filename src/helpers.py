import numpy as np
from  scipy import ndimage, ndarray
import skimage.measure
import cv2
import math

# Numerically stable sigmoid function. Used for output of tree
def sigmoid(x):
    if x < 0:
        return 1 - 1/(1 + math.exp(x))
    else:
        return 1/(1 + math.exp(-x))

# Extract the pixel values from an ellipse shape cut in the image
def extract_ellipse(pixels, center_x, center_y, agg_width, agg_height):   
    # Create an empty (black = 0) nparray same shape as the image, then make a ellipse cut in this array (=1)
    mask = np.zeros_like(pixels)
    mask = cv2.ellipse(mask, center=(center_x, center_y), axes=(agg_width, agg_height), angle=0, startAngle=0, endAngle=360, color=(1), thickness=-1)

    # Apply the cutout to original image, only caring about the non zero parts of mask (as these are the pixels in the ellipse)
    return pixels[mask > 0]

def extract_window(image, shape, x, y, w, h):
    shape = shape.value
    pixels = image.pixels
    dimensions = pixels.shape
    
    # Image is in row, col order
    img_width = dimensions[1]
    img_height = dimensions[0]
    
    # Convert to integers so we can use for indexing
    x_start = int(x * img_width)
    y_start = int(y * img_height)
    agg_width = int(w  * img_width)
    agg_height = int(h * img_height)
    
    # Ensure we are within the images bounds
    x_end = min(x_start + agg_width, img_width)
    y_end = min(y_start + agg_height, img_height)
    
    values = None
    
    # Need to extract different regions based off the shape
    if shape == "Rectangle":
        values = pixels[y_start:y_end, x_start:x_end]
    elif shape == "Column":
        values = pixels[y_start:y_end, x_start:min(x_start+1, img_width)]
    elif shape == "Row":
        values = pixels[y_start:min(y_start+1, img_height), x_start:x_end]
    elif shape == "Ellipse":
        values = extract_ellipse(pixels, x_start, y_start, agg_width, agg_height)
    else:
        print("Shape not found!!")
    
    # Convert to a 1d array
    if values is not None:
        values.flatten()
        
    return values

# Convolve takes an image and a filter to apply, and returns the convolved image (with RELU applied). Uses zero padding for the convolution
def convolve(image, kernel, filter_size):
    # Currently kernel values is a list but we want a 2d array
    kernel = np.asarray(kernel.values).reshape((filter_size, filter_size))

    # Use the constant mode, and padding value of zero to deal with edges of image. Also apply 
    convolved = ndimage.filters.convolve(image.pixels, kernel, mode='constant', cval=0.0)

    # ReLU
    return np.maximum(0, convolved)

# Pooling takes an image, and returns a subsampled version. Uses max pooling of the specified size
def pooling(image, pooling_size):
    return skimage.measure.block_reduce(image.pixels, (pooling_size, pooling_size), np.max)

# Take a shape/region of the image and apply the given function to this region
def agg(fn, image, shape, x, y, width, height):
    # They are integers, treat them as floats instead when passing through
    window = extract_window(image, shape, x.value, y.value, width.value, height.value)
    return fn(window) if window.size > 0 else 0

def protectedDiv(num, den):
    try:
        return num / den
    except ZeroDivisionError:
        return 0.
