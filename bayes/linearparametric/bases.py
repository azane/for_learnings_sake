import numpy as np

#<Localized>
def gaussianType(x,u,v):
    """A gaussian type basis function
        This takes parameters:
            'mean', the locality of the curve
            'variance', the spread of the curve.
        Note that this is not a normal distribution, and is thus not normalized.
    """
    
    return np.exp(-((x-u)**2)/(v**2))
    
#</Localized>

#<Periodic>

#</Periodic>

#<Standard>

#</Standard>