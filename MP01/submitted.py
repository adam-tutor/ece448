'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    n = len(texts)
    word_counter = {}
    for text in texts:
        count = text.count(word0)
        if count in word_counter:
            word_counter[count] += 1
        else:
            word_counter[count] = 1
    max_count = max(word_counter.keys())
    Pmarginal = np.zeros(max_count + 1)
    
    for count, freq in word_counter.items():
        Pmarginal[count] = freq/len(texts)

    #raise RuntimeError("You need to write this part!")
    return Pmarginal

def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''

    counts_word0 = [text.count(word0) for text in texts]
    counts_word1 = [text.count(word1) for text in texts]
    max_count_word0 = max(counts_word0)
    max_count_word1 = max(counts_word1)
    Pcond = np.full((max_count_word0 + 1, max_count_word1 + 1), np.nan)

    for x0 in range(max_count_word0 + 1):
        P_X0_x0 = np.sum(np.array(counts_word0) == x0) / len(texts)

        if P_X0_x0 > 0:
            for x1 in range(max_count_word1 + 1):
                P_X1_given_X0 = np.sum(np.logical_and(np.array(counts_word0) == x0, np.array(counts_word1) == x1)) / np.sum(np.array(counts_word0) == x0)
                Pcond[x0, x1] = P_X1_given_X0

    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    cX0 = len(Pmarginal)
    cX1 = Pcond.shape[1]
    Pjoint = np.zeros((cX0, cX1))

    for x0 in range(cX0):
        for x1 in range(cX1):
            if np.isnan(Pcond[x0, x1]):
                Pjoint[x0, x1] = 0
            else:
                Pjoint[x0, x1] = Pmarginal[x0] * Pcond[x0, x1]

    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    cX0, cX1 = Pjoint.shape
    mu = np.zeros(2)

    for x0 in range(cX0):
        for x1 in range(cX1):
            mu[0] += x0 * Pjoint[x0, x1]
            mu[1] += x1 * Pjoint[x0, x1]

    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    cX0, cX1 = Pjoint.shape
    X0_values, X1_values = np.arange(cX0), np.arange(cX1)

    Sigma = np.zeros((2, 2))

    for x0 in X0_values:
        for x1 in X1_values:
            P_X0_X1 = Pjoint[x0, x1]
            if not np.isnan(P_X0_X1):
                Sigma[0, 0] += (x0 - mu[0]) * (x0 - mu[0]) * P_X0_X1
                Sigma[1, 1] += (x1 - mu[1]) * (x1 - mu[1]) * P_X0_X1
                Sigma[0, 1] += (x0 - mu[0]) * (x1 - mu[1]) * P_X0_X1
                Sigma[1, 0] += (x1 - mu[1]) * (x0 - mu[0]) * P_X0_X1

    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    cX0, cX1 = Pjoint.shape
    X0_values, X1_values = np.arange(cX0), np.arange(cX1)

    Pfunc = Counter()

    for x0 in X0_values:
        for x1 in X1_values:
            P_X0_X1 = Pjoint[x0, x1]
            if not np.isnan(P_X0_X1):
                z = f(x0, x1)
                Pfunc[z] += P_X0_X1

    return Pfunc
    
