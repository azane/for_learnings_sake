import numpy as np

#TODO find better alternative to inverting the covariance matrix every time.
#       save the inverse in memory?
#       just save the determinant for a smaller overhead, but still significant time savings?
#       something about cholesky?

class BayesLinear(object):
    def __init__(self, phiBasis, phiConstants, priorStdDev=10, noiseStdDev=0.2):
        
        #the basis functions for which the parameters are coefficients.
        self._phiBasis = phiBasis
        #the constants applied to the basis function; this differentiates the otherwise identical basis functions.
        self._phiConstants = phiConstants
        
        #the number of parameters, inferred from the number of phiConstant sets
        self._numParams = len(self._phiConstants)
        
        
        #the covariance matrix of the initial prior
        #   just an isotropic prior times the precision
        self.covariance = np.identity(self._numParams)*(1./priorStdDev**2)
        
        #the mean of the initial prior
        #   just a zero mean
        self.mean = np.zeros(self._numParams)
        
        
        #the noise of the data itself.
        #TODO make a learned parameter.
        #   precision is the inverse variance, variance is the standard deviation squared.
        self.noisePrecision = 1/(noiseStdDev**2)
        
        
    def train(self, x, t):
        """Trains the model by updating the posterior over the parameters, given new data.
        """
        #x.shape == (s,d)
        #   where d is the dimensionality of x
        #t.shape == (s,)
        
        self._verify_x(x)
        if t.ndim != 1:
            raise ValueError("The t array must be of rank 1: Sample size columns by 1.")
        if x.shape[0] != t.shape[0]:
            raise ValueError("The x and t sample size do not match.")
        
        #pre-processing
        priorCovarianceInverse = np.linalg.inv(self.covariance)
        fullPhi = self._full_phi(x)  # a matrix of size (s,m), where m is the dimensionality of phi space
        
        #update the posterior over the parameters, using the current distribution as the prior.
        self.covariance = np.linalg.inv(priorCovarianceInverse + self.noisePrecision*np.dot(fullPhi.T, fullPhi))
        self.mean = np.dot(self.covariance, np.dot(priorCovarianceInverse, self.mean) + self.noisePrecision*np.dot(fullPhi.T, t))
        
    def _full_phi(self, x):
        """Maps the input, x, to phi space.
            - where each dimension in phi corresponds to a term in the equation linear in the parameters.
        """
        #x.shape == (s,d)
        
        #map the inputs to the phi space, using the constants to differentiate each dimension (term) in phi
        phiMapped = [self._phiBasis(x, **self._phiConstants[i]) for i in range(self._numParams)]
        
        #phiMapped.shape == (s,m)
        #   s is the number of x vectors
        #   m is the dimensionality of phi space
        return np.array(phiMapped).T
        
    def _verify_x(self, x):
        """A simple verification method verifying the rank of the independent data.
        """
        #x.shape == (s,d)
        if x.ndim != 2:
            raise ValueError("The x array must be of rank 2: Sample size rows by x-dimensionality columns.")
        
    def predictive(self, x):
        """Maps the parametric distribution (in phi space) to the data space and returns the mean and variance of a normal for a given x.
            - where the mappings of both are functions of x.
        """
        self._verify_x(x)
        
        fullPhi = self._full_phi(x)  # .shape == (s,m)
        
        #FIXME the fullPhi is an array of vectors over which the covariance matrix should be broadcast, not just dot producted.
        #       i.e. we need the second term to reduce to a scalar output.
        pVariance = 1/self.noisePrecision + np.dot(fullPhi, self.covariance).dot(fullPhi.T)
        pMean = np.dot(self.mean, fullPhi.T)
        
        return pMean, pVariance
        
    def sample(self, x):
        """Samples the predictive distribution (in data space) for each point in x.
        """
        m, v = self.predictive(x)
        return np.random.normal(m, np.sqrt(v))