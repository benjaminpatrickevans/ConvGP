from deap import gp
import helpers
import autograd.numpy as np
import autograd.scipy.signal
from copy import deepcopy
from autograd import grad
from multiprocessing import Pool
from functools import partial
from sklearn.utils import shuffle


def compile(code, context, arguments):
    if len(arguments) > 0:
        args = ",".join(arg for arg in arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("Gradient Descent : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort.", traceback)


def one_example_loss(pair, tree, filters, classes):
    x, y = pair
    real_class = 1. if y == classes[0] else 0.
    out = tree(x, *filters)[0]
    predicted_class = helpers.sigmoid(out) # Pass through a sigmoid, since treating as probability value
    loss = helpers.ce_loss(real_class, predicted_class)
    return loss

def compute_loss(tree, xs, ys, filters, classes):
    length = len(xs)

    # Need to pass in the tree, filters and classes
    partial_func = partial(one_example_loss, tree=tree, filters=filters, classes=classes)

    # Now pass in each of the images to the function, using map so we can parralelize computation
    losses = map(partial_func, zip(xs, ys))
    total_loss = sum(losses)
    
    # Average loss so batch size doesnt effect magnitude
    return total_loss / float(length)


# Replace all filters with placeholders - So we can pass them in as arguments for computing derivatives easily
def replace_filters_with_args(tree_str, labels, context, arguments):
    filters = {} # Keep a copy of the original filters 
    prefix = "Filter"
    
    for node, label in labels.items():
        # Filters are represented as lists, so we only care for lists
        if type(label) == list:
            arg_name = prefix + str(node)
            filters[node] = label # Save the original value
            labels[node] = arg_name # Replace the value
            arguments.append(arg_name) # Add to the list of arguments, rpefix with filter
    
    # Replace all the filters in the tree with their placeholders
    for arg, value in filters.items():
        tree_str = tree_str.replace(str(value), prefix + str(arg), 1) # Replace only one occurence, in case two filters have same value

    callable_tree = compile(tree_str, context, arguments)
    
    return callable_tree, filters

# Creates a generator of batches
def make_batches(x, y, batch_size):
    length = len(x)
    for i in range(length//batch_size):
        yield x[batch_size*i:batch_size*(i+1)], y[batch_size*i:batch_size*(i+1)]

def split_and_shuffle(x, y, split=0.8):
    x, y = shuffle(x, y)
    length = int(len(x) * split)

    return x[:length], y[:length], x[length:], y[length:]

# Takes a string representation of a tree, and performs gradient descent on the parameters
def gradient_descent(original_tree, xs, ys, context, arguments, classes, epochs, lr):
    tree = deepcopy(original_tree) # Dont modify original tree
        
    tree_str = str(tree)

    # If theres no convolutions, dont bother performing gradient descent as theres no updateable params
    if "Conv" not in tree_str:
        return tree

    _, _, labels = gp.graph(tree)

    arguments = list(arguments) # Make a copy of the arguments so we dont modify original args

    callable_tree, original_filters = replace_filters_with_args(tree_str, labels, context, arguments)

    keys = sorted(list(original_filters.keys())) # Make sure the order is correct, should be ascending
    filters = [original_filters[key] for key in keys] # Retrieve the original filter values

    #Split into validation and training sets
    trainX, trainY, validX, validY = split_and_shuffle(xs, ys)
    batch_size = len(trainX) // 10

    # Whether or not to decay learning rate
    decay = True

    # Flag for when to exit
    finished = False

    # Start with largest loss
    lowest_loss = float("inf") 
    best_filters = list(filters)

    # Number of iterations without improvement to wait before exiting
    patience = epochs // 5 #20%
    num_iterations_without_improv = 0

    last_loss = lowest_loss

    # Gradient descent
    for i in range(epochs):

        # Learning rate decay, halve every 10 epochs
        if i > 0 and i % 10 == 0:
            lr /= 2

        for batchXs, batchYs in make_batches(trainX, trainY, batch_size):
            # Compute changes
            grad_fn = grad(compute_loss, 3) # Compute derivative w.r.t the filters - Third index is the filters, see line below
            deltas = grad_fn(callable_tree, batchXs, batchYs, filters, classes)
            deltas = np.asarray(deltas)

            # Apply changes
            filters -= (lr * deltas)

            # Calculate loss on validation set
            validation_loss = compute_loss(callable_tree, validX, validY, filters, classes)

            # If this is the lowest loss weve seen, save the filters
            if validation_loss < lowest_loss:
                lowest_loss = validation_loss
                best_filters = list(filters)

            # If we are getting worse on the validation set, record how many epochs this has occured
            if validation_loss > last_loss:
                num_iterations_without_improv += 1
            else:
                num_iterations_without_improv = 0

            # if we havent improved for patience epochs, then exit
            if num_iterations_without_improv >= patience:
                filters = best_filters
                finished = True
                break

            last_loss = validation_loss
            
        if finished:
            break


    # Need to modify the tree to have the new filter values - Need to do it in the this fashion to preserve the order
    for idx, key in enumerate(keys):      
        tree[key] = deepcopy(tree[key]) # For some reason deep copy doesnt go DEEP, so need to do it again
        tree[key].value = list(filters[idx])

    return tree