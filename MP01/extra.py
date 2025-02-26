import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    cX = len(PX)
    meanX = np.dot(np.arange(cX), PX)
    p = 1 / (1 + meanX)
    PY = p * (1 - p)**np.arange(cX)

    return p, PY
