"""Pseudo-code for the compression by criterion algorithm
    1. Initialize data space
    2. Initialize compression input locals
    3. Initialize gaussian compression importance average
    4. Sample input locals, mu and lambda, from uniform, or a wide gaussian even.
    5. Determine weight likelihood parameters given the sampled mu and lambda
    6. Sample an x from input locals as a mixture of gaussians, get actual t from data space
    7. Determine likelihood of actual t, given x, sample input locals, and determined weight parameters.
    8. Determine criterion value for mu, lambda, and weight parameters
    9. Update the gaussian compression importance average given the sampled x with likelihood*criterion.
        a. affect neighbors according to CIA length scale
    10. Gravitate all GCIA's toward the prior according to the CIA variance over time.
    11. Fit the input locals given the new averages.
        a. maybe just use MLE to take advantage of easy/incremental gradient ascent

"""
import numpy as np
from ../bayes.clustering.parametric.analytic import SingleGauss

class ComplexityAllocationAverage(SingleGauss):
    """This class provides a set of single gaussians that represent an incrementally updated estimate of a curve.
        The distance/relevance of new info, given input locality is considered for each estimation point in input space.
        The curve gravitates toward its prior at a speed according to variance over time; a lack of relevant data means it will fall toward its prior.
        
        This class #TODO inherits from a class providing single gaussian clustering functionality.
    
    """
    def __init__(self, **kwargs):
        
        super(ComplexityAllocationAverage, self).__init__(**kwargs)

class CriterionCompression(object):
    def __init__(self, **kwargs):
        
        super(CriterionCompression, self).__init__(**kwargs)