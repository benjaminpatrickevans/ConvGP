
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
import cv2
from glob import glob
import pygraphviz as pgv
from autograd import grad
import datetime
import deapfix
import helpers
import evolution
from enum import Enum


# Custom classes for Strongly typed GP structure.
class Image:
    def __init__(self, pixels):
        self.pixels = pixels
        
    def __str__(self):
        return "Image("+repr(self.pixels)+")"
    
    __repr__ = __str__

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


class Filter:
    def __init__(self, values):
        self.values = values

    def __str__(self):
        return "Filter("+repr(self.values)+")"
    
    __repr__ = __str__

class Filter:
    def __init__(self, values):
        self.values = values

    def __str__(self):
        return "Filter("+repr(self.values)+")"
    
    __repr__ = __str__

class Shape:

    allowable_shapes = ["Rectangle", "Ellipse", "Column", "Row"]

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
        pop_size = 1024, generations=50, tourn_size = 7, num_best = 1, crs_rate = 0.75, mut_rate = 0.2):

        self.pooling_size = pooling_size
        self.filter_size = filter_size
        self.pop_size = pop_size
        self.generations = generations
        self.tourn_size = tourn_size
        self.num_best = num_best
        self.crs_rate = crs_rate
        self.mut_rate = mut_rate

        self.pset = self.create_pset()
        self.mstats = self.create_stats()
        self.toolbox = self.create_toolbox()

    # Chooses the appropriate class based on probability value (class[0] if > 0.5, else class 1). Defined here so can be used from outside of class in a consistent way
    def determine_class(self, value):
        return self.classes_[0] if helpers.sigmoid(value) > 0.5 else self.classes_[1]

    # Use the tree to predict class labels
    def predict_labels(self, individual, data):
        # Transform the tree expression in a callable function
        tree = self.toolbox.compile(expr=individual)

        predicted_labels = [self.determine_class(tree(Image(image))) for image in data]
        
        return np.asarray(predicted_labels)

    def predict_probabilities(self, individual, data):
        tree = self.toolbox.compile(expr=individual)

        predicted_labels = []
        for image in data:
            out = tree(Image(image))

            # DEBUGGING - Delete
            if math.isinf(out) or math.isnan(out):
                if not hasattr(self, 'tree'):
                    self.tree = individual
                print(out)
                print(individual)

            # Probability for the two classes - Since binary
            zero_probability = helpers.sigmoid(out)
            one_probability = 1 - zero_probability
            predicted_labels.append([zero_probability, one_probability])
        
        return np.asarray(predicted_labels)   

    # How should fitness of an individual be determined? In this case use classification accuracy
    def fitness_function(self, individual, data, real_labels):            
        # Use the given individual to predict the class labels
        #predicted_labels = self.predict_labels(individual, data)

        # Percentage of elementwise matches between real and predicted labels
        #classification_accuracy = metrics.accuracy_score(real_labels, predicted_labels)

        predictions = self.predict_probabilities(individual, data)
        loss = metrics.log_loss(real_labels, predictions)

        # Deap requires multiple values be returned, so the comma is important!
        return loss, 


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
        pop, log = evolution.gradientEvolution(pop, self.toolbox, self.crs_rate, self.mut_rate, self.generations, stats=self.mstats, halloffame=hof, verbose=verbose)
        
        # Save the results
        self.logbook = log
        self.tree = hof[0]
        self.callable_tree = self.toolbox.compile(expr=self.tree)

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
        pset = gp.PrimitiveSetTyped("MAIN", [Image], float)

        # Need to add the custom types, so deap is able to compile these
        pset.context["Size"] = Size
        pset.context["Position"] = Position
        pset.context["Filter"] = Filter
        pset.context["Shape"] = Shape

        # ================
        # Terminal set
        # ================

        # Convolution Tier
        pset.addEphemeralConstant("Filter", lambda: Filter([random.randint(-3, 3) for _ in range(self.filter_size * self.filter_size)]) , Filter)

        # Aggregation tier
        pset.addEphemeralConstant("Shape", lambda: Shape(random.choice(Shape.allowable_shapes)), Shape) # Shape of window
        
        pset.addEphemeralConstant("Size", lambda: Size(random.uniform(0.15, 0.75)), Size)

        pset.addEphemeralConstant("Pos", lambda: Position(random.uniform(0.05, 0.90)), Position) # Size and position of window

        # Classification Tier
        pset.addEphemeralConstant("Random", lambda: random.uniform(-1, 1), float)

        # ================
        # Function set
        # ================

        # Convolution Tier
        pset.addPrimitive(lambda image, kernel: Image(helpers.convolve(image, kernel, self.filter_size)), [Image, Filter], Image, name="Convolution")
        pset.addPrimitive(lambda image: Image(helpers.pooling(image, self.pooling_size)), [Image], Image, name="Pooling")

        # Aggregation Tier - The inputs correspond to: Image, Shape, X, Y, Width, Height
        pset.addPrimitive(lambda *args: helpers.agg(np.min, *args), [Image, Shape, Position, Position, Size, Size], float, name="aggmin")
        pset.addPrimitive(lambda *args: helpers.agg(np.mean, *args), [Image, Shape, Position, Position, Size, Size], float, name="aggmean")
        pset.addPrimitive(lambda *args: helpers.agg(np.max, *args), [Image, Shape, Position, Position, Size, Size], float, name="aggmax")
        pset.addPrimitive(lambda *args: helpers.agg(np.std, *args), [Image, Shape, Position, Position, Size, Size], float, name="aggstd")

        # Classification Tier - The basic arithmetic operators
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(helpers.protectedDiv, [float, float], float, name="div")
        
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

        # Ramped Half and half generation (full and grow
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
            n.attr["label"] = labels[i]

        g.draw(path)



