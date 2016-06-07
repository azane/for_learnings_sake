"""This file provides a class used for incrementally updating a single gaussian inference problem.
    The functionality 'setting apart' this class, is that includes a method that flattens the parameter distributions, while retaining the
    predictive distribution. This allows simple moving average calculation without actually hanging on to any old points.
    This adds a small restriction, in that the Gamma factor of the distribution must be convex.
"""

import numpy as np
from scipy import stats as spstats

def testSingleGauss(iters):
    
    sg = SingleGauss(1., 0.1, 0., 1.)
    
    for i in range(iters):
        mu = np.random.uniform(-100., 100.)
        var = np.random.uniform(0.00001, 10.)
        
        sg.alpha = np.random.uniform(0.00001, 10.)
        sg.beta = np.random.uniform(0.00001, 10.)
        sg.mup = np.random.uniform(-100., 100.)
        sg.scale = np.random.uniform(0.1, 10.)
        
        ssize = np.random.randint(1,10)
        
        for i in range(10):
            data = np.random.normal(mu, var, ssize)
            sg.fit(data)
            print "mu - mode: " + str(mu - sg.mup)
            print "var - mode: " + str(var - (sg.alpha - 1)/sg.beta)  # assumes convexivity
        
        print " ------------------------------------------------ "

class SingleGauss(object):
    
    def __init__(self, alpha, beta, mup, scale, **kwargs):
        
        try:
            #the parameters for the variance prior
            self.alpha = alpha*1.
            self.beta = beta*1.
            
            #the parameters for the mean prior
            self.mup = mup*1.
            self.scale = scale*1.
            
        except ValueError:
            raise ValueError("Parameters must be convertible to floats.")
        
        assert alpha > 1.
        assert beta > 0.
        assert scale > 0.
        
        
        super(SingleGauss, self).__init__(**kwargs)
    
    def flatten(self, dscale=None):
        """Flattens out distribution
            by reducing scale and alpha arbitrarily, and determining the beta value that will result in the gamma distribution spreading around
            its mode. This actually does not change the predictive distribution, but flattens the parametric distributions so they can
            adapt to new information.
            see https://www.desmos.com/calculator/wmghev4n96 for a visualization
        """
        
        #if it's default, flatten maximally.
        if dscale == None:
            dscale = -2.*(1. - self.alpha) - 1
            
        
        #prevent divide by zero, and maintain convexivity
        assert self.alpha > 1.
        
        dalpha = dscale/2.
        
        #set beta so that the distribution stays centered on its current mode, but spreads its tail in the positive direciton.
        dbeta = self.beta*(1. - (self.alpha - dalpha - 1.)/(self.alpha - 1.))
        
        assert self.scale > dscale
        assert dscale < -2.*(1. - self.alpha)  # restrict the reduction so beta is guaranteed to reduce.
        assert dscale > 0
        assert self.beta > dbeta  # make sure beta will not be less than 0
        assert dbeta > 0  # make sure beta will actually be reduced.
        assert self.alpha > (dalpha + 1.)  # ensure that alpha will not fall below 1, see above
        assert dalpha > 0
        
        self.beta -= dbeta
        self.alpha -= dalpha
        self.scale -= dscale
    
    def fit(self, data):
        """Calculate the posterior parameters given the data, and set the running prior.
            data should be a numpy array vector of 1 dimensional x inputs.
            
            calculations verified with https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        """
        
        assert data.ndim == 1
        
        datamean = data.mean()
        
        self.alpha = self.alpha + data.size/2.
        
        self.beta = self.beta + .5*((data - datamean)**2).sum() \
                    + (self.scale*data.size*(datamean - self.mup)**2)/(2*(self.scale + data.size))
        
        self.mup = (self.scale*self.mup + data.size*datamean)/(self.scale + data.size)
        
        self.scale = self.scale + data.size
        
    def variance(self):
        """Compute and return the predictive variance.
        """
        return ((self.alpha*self.scale)/(self.beta*(self.scale + 1)))**-1
    
    def predict(self):
        """Return a frozen scipy pdf of student-t, which will be the predictive distribution for this data
        """
        
        return spstats.t(2*self.alpha, self.mup, self.variance())
        
if __name__ == "__main__":
    testSingleGauss(100)