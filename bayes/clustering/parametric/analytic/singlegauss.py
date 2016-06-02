"""This file provides a class used for incrementally updating single gaussian 'clustering' problem.
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
        
        assert alpha > 0
        assert beta > 0
        assert scale > 0
        
        
        super(SingleGauss, self).__init__(**kwargs)
    
    def reign_beta(self, dscale):
        """Flattens out distribution
            by reducing scale and alpha arbitrarily, and determining the beta value that will result in an unchanged
            predictive distribution (mean and variance). But since the predictive mean does not depend on anything,
            we need only hold the predictive variance constant.
            in this method, we reduce alpha by half of the scale, because this is the rate at which it increases
            effectively, we are removing dscale points from the distributions memory, widening it, and making it more flexible.
        """
        
        #FIXME this is currently shrinking the predictive variance, need to check math again.
        
        assert dscale > 0 and dscale < self.scale
        
        dalpha = dscale/2
        
        self.beta = self.beta - (self.beta*(self.scale + 1)*(self.scale - dscale))/(self.scale*(self.scale - dscale + 1))
        self.alpha = self.alpha - dalpha
        self.scale = self.scale - dscale
    
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