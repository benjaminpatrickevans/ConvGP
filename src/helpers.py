import autograd.numpy as np
from  scipy import ndimage, ndarray
from autograd import grad
from autograd.scipy import signal
from autograd.scipy.special import expit
from autograd.extend import defvjp
import autograd.numpy.numpy_vjps as vjps
import skimage.measure

class OperatorOut:
    def __init__(self, value, features):
        self.value = value
        self.features = features

    def __str__(self):
        return "Size("+repr(self.value)+")"
    
    __repr__ = __str__

# Numerically stable sigmoid function. Used for output of tree
def sigmoid(x):
    return expit(x)

# Performs arithmetic operator but also needs to deal with pairs as inputs not just floats
def arithmetic_op(fn, l, r):
    # First item in tuple is the value, second is the list of features
    l_value, l_features = l
    r_value, r_features = r
    out = fn(l_value, r_value)

    return (out, l_features + r_features)


def extract_ellipse(pixels, start_x, start_y, agg_width, agg_height):  
    # Top left was passed in, but its nicer for computing if we just use the centre instead
    x_radius = agg_width // 2
    center_x = start_x + x_radius       
    y_radius = agg_height // 2  # y center, half height                            
    center_y = start_y + y_radius

    # Protect against any divide by zeros, by treating the radius as 1 instead
    if x_radius == 0:
        x_radius = 1

    if y_radius == 0:
        y_radius = 1                          

    # This formula will give a true value for all points within ellipse, false for those not in ellipse (this is a basic ellipse with no rotation)
    ellipse = [((x-center_x)/x_radius)**2 + ((y-center_y)/y_radius)**2 <= 1 for (y, x), _ in np.ndenumerate(pixels)]
    ellipse = np.reshape(ellipse, pixels.shape) # Back to a nd array rather than list
     
    return pixels[ellipse]


def extract_window(pixels, shape, x, y, w, h):
    shape = shape.value
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

# Convolve takes an image and a filter to apply, and returns the convolved image (with RELU applied).
def convolve(image, kernel, filter_size):
    # Currently kernel values is a list but we want a 2d array
    kernel = np.reshape(kernel, (filter_size, filter_size))

    # TODO: APPLY PADDING
    convolved = signal.convolve(image, kernel, mode='valid')

    # ReLU
    activated = np.maximum(0, convolved)

    return activated

# Pooling takes an image, and returns a subsampled version. Uses max pooling of the specified size
def pooling(image, size):
    m,n = image.shape
    
    # NOTE: This has only been tested with size =2, likely need changes with different pooling sizes 
    if m % size != 0:
        image = image[:-1,:]
    
    if n % size != 0:
        image = image[:,:-1]
        
    m,n = image.shape
        
    out = image.reshape(m//size,size,n//size,size).max(axis=(1,3))
    return out

# Take a shape/region of the image and apply the given function to this region
def agg(fn, image, shape, x, y, width, height):
    # They are integers, treat them as floats instead when passing through
    window = extract_window(image, shape, x.value, y.value, width.value, height.value)
    out = fn(window) if window.size > 0 else 0
    return (out, [out])


def grad_np_mean(ans, x, axis=None, keepdims=False):
    shape, dtype = anp.shape(x), anp.result_type(x)
    def vjp(g):
        g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        return g_repeated / num_reps
    return vjp
# Important - Need to redefine the gradient forn standard deviation as it will cause nans if the std is 0. 
#We can just use the means gradient, as the magnitude of each value is not hugely important with gradient descent (rather the direction)
defvjp(np.std, vjps.grad_np_mean)

def protectedDiv(num, den):
    if den != 0:
        return num / den
    else:
        return 0.

def mse_loss(real_label, predicted_label):
    return 1/2 * (real_label - predicted_label)**2

def ce_loss_old(real_label, predicted_label):
    # Just to be safe so we dont get any log zeros
    predicted_label = bound_check(predicted_label)
    return - np.sum(np.multiply(real_label, np.log(predicted_label)) + np.multiply((1-real_label), np.log(1-predicted_label)))

# Works for only a single label input
def ce_loss(real_label, predicted_label):
    # Just to be safe so we dont get any log zeros
    predicted_label = bound_check(predicted_label)
    class_zero = real_label * np.log(predicted_label) # Need to use np.log for autograd to work rather than math.log
    class_one = (1-real_label) * np.log(1-predicted_label)
    return - (class_zero + class_one)


# Make 1s 0.99999, and 0s 0.000001 - No effect on other inputs
def bound_check(x):
    if x == 1:
        return 0.99999999
    elif x == 0:
        return 0.00000001
    else:
        return x