"""Outline for the compression by criterion algorithm
    1. Initialize data space
    2. Initialize compression input locals
    3. Initialize gaussian compression importance averages
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
import tensorflow as tf
from ../bayes.clustering.parametric.analytic import SingleGauss


class ComplexityAllocationAverage(object):
    """Provides a set of SingleGauss objects that will track the complexity allocation estimations throughout the data space.
    """
    def __init__(self, **kwargs):
        
        super(ComplexityAllocationAverage, self).__init__(**kwargs)

class InputLocals(object):
    """Provides a set of gaussian basis functions that will morph around the input space to meet the complexity allocation curve.
        Provides input space sampling method that views the input locals as a mixture of gaussians.
    """
    def __init__(self, rBot, rTop, numBases, **kwargs):
        
        #the bottom and top of the range of data over which the compression will take place
        #   these can be array-like if the data space is multidimensional. thus, only rectangles are currently supported.
        try:
            self.rBot = np.array(rBot)
            self.rTop = np.array(rTop)
        except ValueError:
            raise ValueError("rBot and rTop must be array-like.")
        assert self.rBot.ndim == 1
        assert self.rTop.ndom == 1
        assert self.rBot.size == self.rTop.size
        assert (self.rTop > self.rBot).all()
            
        self.numBases = numBases
        
        
        #construct TF graph
        self.graph = tf.graph()
        self.nodeDict = {}
        with self.graph.as_default():
            self._mixture_graph()
            self._error_graph()
        
        super(InputLocals, self).__init__(**kwargs)
        
    def _mixture_graph(self):
        """Builds a graph to compute the value of the summed bases.
        """
        
        #the 'means' and 'covariances' of the basis curves
        _mus = (self.rTop - self.rBot) * np.random.random((self.numBases, self.rBot.size)) + self.rBot  # an array of row vectors denoting means
        mus = tf.Variable(_mus)
        _vardiag = (((self.rTop - self.rBot)/self.numBases)/2)**2 # we want a diagonal, with space/numBases, then divide by two to get to stdev, then square for var
        _cvar = np.diag(vardiag)
        _cvars = np.tile(cvar, (self.numBases, 1, 1))  # an array of covariance matrices, these start identical, and then change independently.
        cvars = tf.Variable(_cvars)
        
        pbases = #TODO sum of mixed multivariate gaussians.
        
        self.nodeDict['mus'] = mus
        self.nodeDict['cvars'] = cvars
        
    def _error_graph(self):
        """Tacks the error function on to the mixture graph, exposes relevant tensors in dict.
        """
        
        
    
    def sample(self, x, size=1):
        """Select 1 of k 'x' points by treating the bases as a gaussian mixture, returning the index of x.
            Because the sampling is discrete, we first get the normalized probabilities of drawing an 'x' in light of the bases distributions and the x value.
            Given the probabilities, we sample a multinomial distribution.
        """
        
        #FIXME pseudo
        unnorm = mixed_bases(x)
        pvals = unnorm/(unnorm.sum())
        #FIXME /pseudo
        
        return np.random.multinomial(xvals.size, pvals, size=size)
        
    def fit(self, xvals, tvals):
        """Take xvals and tvals and feed it into the error gradient graph, apply the gradients to the input locals to reduce the error.
        """
        
        #FIXME pseudo
        step = 1.
        gmus, gcvars = error_grad(xvals=xvals, tvals=tvals)
        self.mus -= gmus*step
        self.cvars -= gcvars*step
        #FIXME /pseudo
    
class CriterionCompression(object):
    """Compresses data according to criterion, allocating constant complexity accordingly.
        Stores data to be compressed.
        Samples from all potential basis functions.
        Uses linearparametric to calculate weights and likelihood.
        Feeds to criterion function.
        Updates ComplexityAllocationAverage
    """
    def __init__(self, **kwargs):
        
        super(CriterionCompression, self).__init__(**kwargs)