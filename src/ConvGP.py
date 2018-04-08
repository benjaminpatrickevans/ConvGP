
# coding: utf-8

# In[ ]:

import scipy as sp
from  scipy import ndimage
import random
from glob import glob
import stgp
from sklearn import neighbors, svm, tree, naive_bayes, ensemble
from sklearn import metrics
from sklearn.utils import shuffle
from skimage import transform
import numpy as np
import search
import time
import sys
import math
import helpers
import os
import pickle

# # Read in the data for training/testing
# 1. Read the images from disk
# 2. Save these in a dict from label -> images
# 3. Split these based off label into training/testing images
# 
# Need to do it in this order to ensure we get an equal split of instances from each class in the data, since classification accuracy is used as fitness this is important.

# In[ ]:

# Reads in all the data as a dict from label -> [images]
def read_data(directory, scale=False, scaled_height=None, scaled_width=None):
    data = {}

    # Assumes the images are in subfolders, where the folder name is the images label
    for subdir in glob(directory+"/*/"):
        label = subdir.split("/")[-2] # Second to last element is thee class/sub folder name
        images = [ndimage.imread(image, flatten=True) for image in glob(subdir+"/*.*")] # Read in all the images from subdirectories. Flatten for greyscale
    
        images = [image.astype(float) / 255. for image in images] # Store in range 0..1 rather than 0..255
        
        if scale: # Resize images
            images = [transform.resize(image, (scaled_height, scaled_width)) for image in images]
            
        # Shuffle the images (seed specified at the top of program so this will be reproducable)
        random.shuffle(images)
        data[label] = images
        
    # Set of all class names
    class_names = list(data.keys())

    # Sanity check
    if len(class_names) != 2:
        print("Binary classification only! But labels found were:", labels)
    
    return data, class_names

# Splits the data into four arrays trainingX, trainingY, testingX, testingY
def format_and_split_data(data, class_names, split, seed):
    trainingX = []
    trainingY = []
    
    testingX = []
    testingY = []
    
    # For all the classes, split into training/testing (need to do it per class to ensure we get a good split of all classes)
    for label in class_names:
        x = data[label]
        length = int(len(x))
        y = [label] * length
        
        training_length = int(length * split)
        trainingX.extend(x[:training_length])
        trainingY.extend(y[:training_length])
        
        testingX.extend(x[training_length:])
        testingY.extend(y[training_length:])
    
    # And just so the order isnt all class1s then all class2s, shuffle the data in unison
    trainingX, trainingY = shuffle(trainingX, trainingY, random_state=seed)
    testingX, testingY = shuffle(testingX, testingY, random_state=seed)

    return trainingX, trainingY, testingX, testingY


# # Run the various models
# 
# Now we have the data, we can run and evaluate the various algorithms

# In[ ]:

def pretty_float(f):
    return "{0:.2f}".format(f)

# The method of comparison
def classification_accuracy(real_labels, predicted_labels):
    return metrics.accuracy_score(real_labels, predicted_labels)

def fit_and_evaluate(model, trainingX, trainingY, testingX, testingY, seed=None, verbose=False):
    start = time.time() # Track the time taken

    if seed is not None:
        model.fit(trainingX, trainingY, seed=seed, verbose=verbose)
    else:
        model.fit(trainingX, trainingY)
    
    training_time = time.time() - start
        
    predicted_training = model.predict(trainingX)
    
    start = time.time()
    predicted_testing = model.predict(testingX)
    testing_time = time.time() - start
    
    return classification_accuracy(trainingY, predicted_training), classification_accuracy(testingY, predicted_testing), training_time, testing_time 



# In[ ]:

def print_stats(title, arr):
    print(title, pretty_float(np.min(arr)), pretty_float(np.mean(arr)), pretty_float(np.max(arr)), pretty_float(np.std(arr)), len(arr))


# In[ ]:

def run_general_classifiers(trainingX, trainingY, testingX, testingY):
    print("General Classifiers")

    # The general classification methods require a list of features, rather than a 2d array so we need to flatten these
    flattened_trainingX = [image.flatten() for image in trainingX]
    flattened_testingX = [image.flatten() for image in testingX]

    # The general classifiers to compare against
    general_classifiers = {
        "Nearest Neighbour": neighbors.KNeighborsClassifier(1),
        "SVM": svm.SVC(),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Adaboost": ensemble.AdaBoostClassifier()
    }

    print("Name, Training accuracy, Testing Accuracy, Training Time, Testing Time")
    
    # These methods are deterministic, so only need to be run once
    for classifier in general_classifiers:
        model = general_classifiers[classifier]
        training_accuracy, testing_accuracy, training_time, testing_time = fit_and_evaluate(model, flattened_trainingX, trainingY, flattened_testingX, testingY)
        print(classifier, pretty_float(training_accuracy * 100), pretty_float(testing_accuracy * 100), pretty_float(training_time * 1000), pretty_float(testing_time * 1000))



# In[ ]:

def run_convgp(trainingX, trainingY, testingX, testingY, evolution_seed, lr, gd_frequency, extended):
    print("ConvGP")
    
    convgp = stgp.ConvGP(lr=lr, gd_frequency=gd_frequency, extended=extended)
    
    # Print out the parameters for reference
    convgp.print_info()
    print("\tEvolution seed", evolution_seed)
        
    training_accuracy, testing_accuracy, training_time, testing_time = fit_and_evaluate(convgp, trainingX, trainingY, testingX, testingY, seed=evolution_seed, verbose=False)

    stats = [training_accuracy * 100, testing_accuracy * 100, training_time, testing_time]
    
    return convgp, stats

def save_stats(stats, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(stats, fp)
    print(stats)

# In[ ]:

def run(dataset_name, training_seed, evolution_seed, lr, gd_frequency, scale=False, extended=False, training_split=0.5):
    # Reproducability for shuffle
    random.seed(training_seed)

    # Which data to use
    data_directory = "data/"

    # Where to save the output
    output_directory = "out/"

    # If the dir doesnt exist, make it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("Data is:", dataset_name)
    print("Seed for data shuffle is:", training_seed)

    # Used only if scale is set to True. Must be used if images are of different sizes
    scaled_width = 64 
    scaled_height = 64

    # Read and split data into training and testing    
    data, class_names = read_data(data_directory+dataset_name, scale, scaled_width, scaled_height)
    trainingX, trainingY, testingX, testingY = format_and_split_data(data, class_names, training_split, training_seed)
    
    #run_general_classifiers(trainingX, trainingY, testingX, testingY)
    convgp, stats = run_convgp(trainingX, trainingY, testingX, testingY, evolution_seed, lr, gd_frequency, extended)

    extended_text = "E" if extended else ""
    lr_text = "-"+str(lr)+"-" if gd_frequency > 0 else ""
    out_prefix = output_directory + dataset_name+"-"+str(training_seed)+"-"+str(evolution_seed)+lr_text+str(gd_frequency)+extended_text

    convgp.save_logbook(out_prefix+"-logbook.txt")
    save_stats(stats, out_prefix+"-stats.txt")
    convgp.save_tree(out_prefix+"-best.png")


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("You must run the program with 8 args: dataset_name (str), training seed (int), evolutionary seed (int), learning rate(float), gradient descent frequency (int), scale (true/false), extended (true/false).")
        print(sys.argv)
        sys.exit()
        
    dataset_name = sys.argv[1]
    training_seed = int(sys.argv[2])
    evolution_seed = int(sys.argv[3])
    lr = float(sys.argv[4])
    gd_frequency = int(sys.argv[5])
    scale = sys.argv[6].upper() == "TRUE"
    extended = sys.argv[7].upper() == "TRUE"
    
    run(dataset_name, training_seed, evolution_seed, lr, gd_frequency, scale, extended)

