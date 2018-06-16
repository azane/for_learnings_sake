import numpy as np


# <Localized>
def gaussianType(x, u, v):
    """A gaussian type basis function
        This takes parameters:
            'mean', the locality of the curve
            'variance', the spread of the curve.
        Note that this is not a normal distribution, and is thus not normalized.
    """
    # x.shape == (s,d)
    # u.shape == (1,d)
    # v.shape == (1,)
    # return shape == (s,)

    return (np.exp(-(np.linalg.norm(x - u, axis=1) ** 2) / (v ** 2)))


# </Localized>

# <Periodic>

# </Periodic>

# <Standard>

def polynomialType(x, p):
    return x ** p
    # </Standard>
