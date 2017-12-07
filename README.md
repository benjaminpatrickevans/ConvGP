# ConvGP

This was my project as part of the requirements for my Honours degree in Computer Science at Victoria University of Wellington. A novel method for Binary image classification.

## How to use?

I set this up to feature a similar API to sklearn. The code can be ran and make predictions in only three lines

```python
gp = stgp.ConvGP()
gp.fit(trainingX, trainingY)
predictions = gp.predict(testingX)
```

Any questions feel free to open an issue

## About:
The key idea is to combine aspects from genetic programming and convolutional neural networks
to overcome various limitations of ConvNets, i.e.

- Need for manually crafted architectures
- Poor Interpretability. Although google brain appears to be doing some promising research in this area [here](https://distill.pub/2017/feature-visualization/), however a large limitation is still interpretability of feature interaction 
- Require large amounts of training data

An example of an evolved architecture is shown below

![Example Tree](res/images/example-tree.png "A sample solution for the JAFFE dataset")

A breakdown of the architecture is given below, a structure is enforced using strongly typed genetic programming.

![Example Architecture](res/images/tier.png "Example tree demonstranting the architecture")

The proposed method overcomes some of the aforementioned problems.

- The architecture is automatically evolved rather than manually crafted
- The solution offers high interpretability, as shown with the example above

## Future Work:
I recieved a summer scholarship working on extending this work, and the main current focus is 

- Update filters through gradient descent (in progress)
- Implementing a new fitness function which can effictevly learn from a small number of instances to overcome the need for large amounts of training data

### Foot note:
This work was originally written in [ECJ](https://cs.gmu.edu/~sean/papers/gecco17-ecj.pdf), and was based on some existing code from my supervisors work on [2TGP](http://www.sciencedirect.com/science/article/pii/S0957417412003867). The code base has since been ported to Python3, which is what you see attached here. This was done for a number of reasons, mainly the large number of available libraries which can reduce the overall code size (as originally code was written all from scratch) and the improved readability.
