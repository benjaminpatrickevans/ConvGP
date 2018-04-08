
# coding: utf-8

# In[1]:

#import numpy as np
import autograd.numpy as np
from  scipy import ndimage, ndarray
import skimage.measure
from sklearn import metrics
from deap import algorithms, base, creator, tools, gp
import random, operator, math
import matplotlib.pyplot as plt
from glob import glob
import pygraphviz as pgv
from autograd import grad
import datetime
import deapfix
import helpers
import evolution
from enum import Enum
import sys
from scoop import futures
import pickle

# Custom classes for Strongly typed GP structure.

class Size:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Size("+repr(self.value)+")"
    
    __repr__ = __str__

class Position:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Position("+repr(self.value)+")"
    
    __repr__ = __str__


class Shape:

    allowable_shapes = {"Rectangle", "Ellipse", "Column", "Row"}

    def __init__(self, value):
        if value not in Shape.allowable_shapes:
            raise Exception("Invalid shape: ", value)
        self.value = value

    def __str__(self):
        return "Shape("+repr(self.value)+")"
    
    __repr__ = __str__


class ConvGP():

    """ Classifier implementing the ConvGP method. Implemented using the DEAP library. """

    def __init__(self, pooling_size=2, filter_size=3,
        pop_size = 1024, generations=50, tourn_size = 7, num_best = 5, crs_rate = 0.75, mut_rate = 0.2, gd_frequency=10, epochs=100, lr=0.05, extended=False):

        self.pooling_size = pooling_size
        self.filter_size = filter_size
        self.pop_size = pop_size
        self.generations = generations
        self.tourn_size = tourn_size
        self.num_best = num_best
        self.crs_rate = crs_rate
        self.mut_rate = mut_rate

        self.gd_frequency = gd_frequency # When to run Gradient descent, 5 means every 5 epochs. Set to -1 to disable gradient descent

        self.epochs = epochs
        self.lr = lr
        self.extended = extended # Whether or not to apply gradient descent for extended period on final generation

        self.pset = self.create_pset()
        self.mstats = self.create_stats()
        self.toolbox = self.create_toolbox()

    # Prints out the parameters for reference
    def print_info(self):
        print("ConvGP Settings")
        print("\tPooling size:", self.pooling_size)
        print("\tFilter size:", self.filter_size)
        print("\tPopulation size:", self.pop_size)
        print("\tTournament size:", self.tourn_size)
        print("\tGenerations:", self.generations)

        print("\tCrossover rate:", self.crs_rate)
        print("\tMutation rate:", self.mut_rate)
        print("\tReproduction rate:", 1 - self.mut_rate - self.crs_rate)

        print("\tGradient descent frequency:", self.gd_frequency)
        print("\tEpochs:", self.epochs)
        print("\tLearning Rate:", self.lr)

    # Chooses the appropriate class based on probability value (class[0] if > 0.5, else class 1). Defined here so can be used from outside of class in a consistent way
    def determine_class(self, value):
        return self.classes_[0] if value > 0.5 else self.classes_[1]

    # Use the tree to predict class labels
    def predict_labels(self, individual, data):
        # Transform the tree expression in a callable function
        tree = self.toolbox.compile(expr=individual)

        predicted_labels = [self.determine_class(helpers.sigmoid(tree(image)[0])) for image in data]
        
        return np.asarray(predicted_labels)

    def predict_probabilities(self, individual, data):
        tree = self.toolbox.compile(expr=individual)

        predicted_labels = []

        for image in data:
            out = tree(image)[0]

            # Probability for the two classes - Since binary
            zero_probability = helpers.sigmoid(out)
            one_probability = 1 - zero_probability
            predicted_labels.append([zero_probability, one_probability])
        
        return np.asarray(predicted_labels)   

    # How should fitness of an individual be determined? In this case use classification accuracy with a penalty
    def fitness_function(self, individual, data, real_labels):  
        predicted_labels = self.predict_labels(individual, data)

        # Percentage of elementwise matches between real and predicted labels
        classification_accuracy = metrics.accuracy_score(real_labels, predicted_labels)

        # Deap requires multiple values be returned, so the comma is important!
        return classification_accuracy, 


    def fit(self, trainingX, trainingY, seed = 1, verbose = True):
        # Reproducability
        random.seed(seed)
        np.random.seed(seed)

        # The possible class labels/names as an alphabetical list. This is first converted to a set to get the unique values. 
        self.classes_ = sorted(set(trainingY))

        if len(self.classes_) != 2:
            raise Exception("This method only supports binary classification! But labels found were:" + self.classes_)

        self.toolbox.register("evaluate", self.fitness_function, data=trainingX, real_labels=trainingY)

        # Run training process
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(self.num_best)
        pop, log = evolution.gradientEvolution(pop, self.toolbox, self.crs_rate, self.mut_rate, self.generations,
            trainingX, trainingY, self.pset.context, self.pset.arguments, self.classes_, self.gd_frequency, self.epochs, self.lr, self.extended, stats=self.mstats, halloffame=hof, verbose=verbose)
        
        # Save the results
        self.logbook = log
        self.hof = hof
        self.tree = hof[0]


    def predict(self, data):
        if self.tree is None:
            raise Exception("You must call fit before predict!")

        # Predict the labels using the saved tree
        return self.predict_labels(self.tree, data)

     # Return probabilities
    def predict_proba(self, data):
        if self.tree is None:
            raise Exception("You must call fit before predict!")

        return self.predict_probabilities(self.tree, data)
        

    def create_pset(self):
        # Program Takes in a single image as input. Outputs a float which is then used for classification by thresholding
        pset = gp.PrimitiveSetTyped("MAIN", [ndarray], tuple)

        # Need to add the custom types, so deap is able to compile these
        pset.context["Size"] = Size
        pset.context["Position"] = Position
        pset.context["Shape"] = Shape

        # ================
        # Terminal set
        # ================

        # Convolution Tier
        pset.addEphemeralConstant("Filter", lambda: [random.uniform(-1, 1) for _ in range(self.filter_size * self.filter_size)] , list)

        # Aggregation tier
        pset.addEphemeralConstant("Shape", lambda: Shape(random.choice(tuple(Shape.allowable_shapes))), Shape) # Shape of window
        
        pset.addEphemeralConstant("Size", lambda: Size(random.uniform(0.15, 0.75)), Size)

        pset.addEphemeralConstant("Pos", lambda: Position(random.uniform(0.05, 0.90)), Position) # Size and position of window

        # Classification Tier
        pset.addEphemeralConstant("Random", lambda: random.uniform(-1, 1), float)

        # To respect half and half generation
        pset.addEphemeralConstant("RandomTuple", lambda: (random.uniform(-1, 1), []), tuple)

        # ================
        # Function set
        # ================

        # Convolution Tier
        pset.addPrimitive(lambda image, kernel: helpers.convolve(image, kernel, self.filter_size), [ndarray, list], ndarray, name="Convolution")
        pset.addPrimitive(lambda image: helpers.pooling(image, self.pooling_size), [ndarray], ndarray, name="Pooling")

        # Aggregation Tier - The inputs correspond to: Image, Shape, X, Y, Width, Height. 
        # The output is a pair containing the output of the aggregation function and the output stored in a list for feature construction purposes
        pset.addPrimitive(lambda *args: helpers.agg(np.min, *args), [ndarray, Shape, Position, Position, Size, Size], tuple, name="aggmin")
        pset.addPrimitive(lambda *args: helpers.agg(np.mean, *args), [ndarray, Shape, Position, Position, Size, Size], tuple, name="aggmean")
        pset.addPrimitive(lambda *args: helpers.agg(np.max, *args), [ndarray, Shape, Position, Position, Size, Size], tuple, name="aggmax")
        pset.addPrimitive(lambda *args: helpers.agg(np.std, *args), [ndarray, Shape, Position, Position, Size, Size], tuple, name="aggstd")

        # Classification Tier - The basic arithmetic operators however they need to take tuples since the features are passed through the tree
        pset.addPrimitive(lambda x, y: helpers.arithmetic_op(operator.add, x, y), [tuple, tuple], tuple, name="add")
        pset.addPrimitive(lambda x, y: helpers.arithmetic_op(operator.sub, x, y), [tuple, tuple], tuple, name="sub")
        pset.addPrimitive(lambda x, y: helpers.arithmetic_op(operator.mul, x, y), [tuple, tuple], tuple, name="mul")
        pset.addPrimitive(lambda x, y: helpers.arithmetic_op(helpers.protectedDiv, x, y), [tuple, tuple], tuple, name="div")
        
        return pset

    def create_stats(self):
        # Data to track per generation. Track the min, mean, std, max of the fitness and tree sizes 
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)

        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        return mstats

    def create_toolbox(self):
        # This a maximization problem, as fitness is classification accuracy (higher the better)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Individuals in the population should be represented as tree structures (standard GP)
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Ramped Half and half generation (full and grow)
        toolbox.register("expr", deapfix.genHalfAndHalf, pset=self.pset, min_=2, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        # Tournament size
        toolbox.register("select", tools.selTournament, tournsize=self.tourn_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", deapfix.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        # Max tree heights for crossover and mutation
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


        return toolbox

    # Used to return a list of constructed features for an image, using the best individual
    def construct_features(self, image):
        if self.tree is None:
            raise Exception("You must call fit before attempting to construct features!")

        callable_tree = self.to_callable(self.tree)

        # 0 is output, 1 is constructed features
        return callable_tree(image)[1]

    # Convert a tree to a callable function
    def to_callable(self, individual):
        return self.toolbox.compile(expr=individual)

    def save_logbook(self, file_name):
        if self.logbook is None:
            raise Exception("You must call fit before save!")

        with open(file_name, 'wb') as fp:
            pickle.dump(self.logbook, fp)

    #  To Plot/draw the resulting trees
    def save_tree(self, file_name):
        if self.tree is None:
            raise Exception("You must call fit before save!")

        nodes, edges, labels = gp.graph(self.tree)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            label = labels[i]
            label_type = type(label)

            # Pretty formatting
            if label_type in [Position, Size]:
                # 2 decimal points
                label = "{0:.2f}".format(label.value)
            elif label_type == Shape:
                label = label.value
            elif label_type == tuple:
                # 2DP, only first part of tuple matter (second part is constructed features)
                label = "{0:.2f}".format(label[0])
            elif label_type == list:
                formatted_list = [ "{0:.2f}".format(elem) for elem in label]
                formatted_list = np.reshape(formatted_list, (3,3))
                label = formatted_list

            n.attr["label"] = label


        g.draw(file_name)



