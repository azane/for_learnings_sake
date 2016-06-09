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
        """Initializes the input locals.
            Processes the data space ranges,
             where rBot and rTop are vectors of equal size dictating the rectangle encompassing the data space.
            rBot[d] is the smallest point on dimension d, and rTop[d] the largest.
            numBases is the number of basis curves composing the set of input locals.
        """
        
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
            This is essentially a multivariate gaussian additive mixture, where the mixing coefficients weight each curve equally.
            
            Due to the complexity of the calculation, halfway calculations are prepended with '_', denoting that they are not an end result.
            Further, all tensors needing to be accessed in the future are added to the node dictionary.
        """
        
        #<Variables>
        #the 'means' and 'covariances' of the basis curves
        _mus = (self.rTop - self.rBot) * np.random.random((self.numBases, self.rBot.size)) + self.rBot  # an array of row vectors denoting means
        mus = tf.Variable(_mus)
        _vardiag = (((self.rTop - self.rBot)/self.numBases)/2)**2 # we want a diagonal, with space/numBases, then divide by two to get to stdev, then square for var
        _cprec = np.diag(vardiag)**-1
        _cprecs = np.tile(_cprec, (self.numBases, 1, 1))  # an array of covariance matrices, these start identical, and then change independently.
        cprecs = tf.Variable(_cprecs)
        #</Variables>
        
        #<Placeholders>
        x = tf.placeholder(tf.float32, shape=[None, self.rBot.size], name='x')  # .shape == (S, D)
        #</Placeholders>
        
        #<MV Gaussian>
        #TODO optimize this process
        #NOTE: the 'mixing coefficients' appear in self.numBases**(-1)
        _pnorms = (self.numBases**(-1) * ((2*np.pi)**(-self.numBases/2.))) * tf.sqrt(tf.batch_matrix_determinant(cprecs))  # .shape == (B,)
        
        _x = tf.expand_dims(x, 0)  # .shape == (1, S, D)
        _mus = tf.expand_dims(mus, 1)  # .shape == (B, 1, D)
        _diff = _x - _mus  # .shape == (B, S, D)
        _difXcov = tf.batch_matmul(_diff, cprecs)  # .shape == (B, S, D) . (B, D, D) == (B, S, D)
        _difXcovXdif = tf.batch_matmul(tf.expand_dims(_difXcov, 2), tf.expand_dims(_diff, 3))  # .shape == (B, S, 1, D) . (B, S, D, 1) == (B, S, 1, 1)
        
        _pexp = tf.exp((-1/2.)*(tf.squeeze(_difXcovXdif)))  # .shape == (B, S)
        
        _bases = (tf.expand_dims(_pnorms, 1) * _pexp)  # .shape == (B, S)
        mixedbases = tf.reduce_sum(_bases, reduction_indices=0)  # .shape == (S,)
        #</MV Gaussian>
        
        self.nodeDict['mus'] = mus
        self.nodeDict['cprecs'] = cprecs
        self.nodeDict['x'] = x
        self.nodeDict['mixedbases'] = mixedbases
        
    def _error_graph(self):
        """Tacks the sum of squares error function on to the mixture graph, exposes relevant tensors in dict.
        """
        
        #<Placeholders>
        t = tf.placeholder(tf.float32, shape=[None], name='t')  # .shape == (S,)
        #</Placeholders>
        
        #<Error>
        #Just the squared difference between the target and the current mixedbases.
        errorcurve = tf.squared_difference(t, self.nodeDict['mixedbases'])  # .shape == (S,)
        error = tf.reduce_sum(errorcurve)
        #</Error>
        
        self.nodeDict['t'] = t
        self.nodeDict['errorcurve'] = errorcurve
        self.nodeDict['error'] = error
    
    def sample(self, x, size=1):
        """Select 1 of k 'x' points by treating the bases as a gaussian mixture, returning the index of x.
            Because the sampling is discrete, we first get the normalized probabilities of drawing an 'x' in light of the bases distributions and the x value.
            Given the probabilities, we sample a multinomial distribution.
        """
        
        with self.graph.as_default():
            
            feedDict = {
                            self.nodeDict['x']: x
                        }
            
            unnorm = self.nodeDict['sess'].run(self.nodeDict['mixedbases'], feed_dict=feedDict)[0]
            pvals = unnorm/(unnorm.sum())
            
            return np.random.multinomial(x.size, pvals, size=size)
        
    def fit(self, xvals, tvals):
        """Take xvals and tvals and feed it into the error gradient graph, apply the gradients to the input locals to reduce the error.
        """
        
        #FIXME pseudo
        #step = 1.
        #gmus, gcvars = error_grad(xvals=xvals, tvals=tvals)
        #self.mus -= gmus*step
        #self.cvars -= gcvars*step
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