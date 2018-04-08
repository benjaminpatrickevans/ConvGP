import random
from inspect import isclass
import sys

# DEAP has some issues with strongly typed GP and tree generation (see here: https://groups.google.com/forum/#!searchin/deap-users/stgp/deap-users/YOeb65eRNG4/AYUMcNldhdwJ and here: https://groups.google.com/forum/#!msg/deap-users/adq50--lzJ4/hefHPJKpBQAJ )
# The following code replaces some of the code in DEAP, to make this work. 

# The block off code below be ignored, essentially its a workaround to stop the need for "identity nodes" which bloat the trees, to fix
# the issue of strongly typed trees not being able to be generated in particular circumstances (i.e. full method but cant happen with current types)

# genHalfAndHalf, genFull and genGrow copied directly from deap (https://github.com/DEAP/deap/blob/master/deap/gp.py)
def genFull(pset, min_, max_, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height
    return generate(pset, min_, max_, condition, type_)

def genGrow(pset, min_, max_, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a a node should be a terminal.
        """
        return depth == height or             (depth >= min_ and random.random() < pset.terminalRatio)
    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)

# Small change made to method below from DEAP version. If you try and add a primtiive, but none of the appropriate type
# is available, then try add a terminal instead.
def generate(pset, min_, max_, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        
        # If we are at the end of a branch, add a terminal
        if condition(height, depth):
            term = add_terminal(pset, type_)
            expr.append(term)
            
        # Otherwise add a function    
        else:
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except IndexError: 
                # This is where the change occurs, if no primitive is available, try and add a terminal instead
                term = add_terminal(pset, type_)
                expr.append(term)
            
    return expr


def add_terminal(pset, type_):
    try:
        term = random.choice(pset.terminals[type_])
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError("The custom generate function tried to add a terminal of type '%s', but there is none available." % (type_,), traceback)
    if isclass(term):
        term = term()
    
    return term