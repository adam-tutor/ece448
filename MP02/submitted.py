'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    
    frequency = {}
    for class_y in train.keys():
        frequency[class_y] = Counter()
        for text in train[class_y]:
            text_range = range(len(text) - 1)
            for idx in text_range:
                word1 = text[idx]
                word2 = text[idx + 1]
                frequency[class_y][word1 + "*-*-*-*" + word2] += 1
    #raise RuntimeError("You need to write this part!")
    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    nonstop = {}
    for class_y in frequency.keys():
            occurrences = frequency[class_y]
            nonstop[class_y] = Counter()
            for word in list(occurrences):
                word1, word2 = word.split("*-*-*-*")
                if(not((word1 in stopwords) and (word2 in stopwords))): #if both tokens are not stopwords, then continue
                    nonstop[class_y][word1 + "*-*-*-*" + word2] = occurrences[word]
    return nonstop
    #raise RuntimeError("You need to write this part!")


def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    for class_y in nonstop:
        likelihood[class_y] = {}
        unique_bigrams = list(nonstop[class_y].keys())
        token_cnt = sum(nonstop[class_y].values())
        for bigram in unique_bigrams:
            denom = (token_cnt + smoothness * (len(unique_bigrams) + 1))
            likelihood[class_y][bigram] = (nonstop[class_y][bigram] + smoothness)/denom
        likelihood[class_y]["OOV"] = smoothness / denom
    return likelihood
    #raise RuntimeError("You need to write this part!")

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    for i in texts:
        p_prob = np.log(prior)
        n_prob = np.log(1 - prior)
        for k in range(len(i) - 1):
            if(not((i[k] in stopwords) and (i[k + 1] in stopwords))): #same setup as remove_stopwords for checking words
                bigram = i[k] + "*-*-*-*" + i[k + 1] #word1 = i[k] and word2 = i[k+1]
                if bigram in likelihood["pos"]:
                    p_prob = p_prob + np.log(likelihood["pos"][bigram])
                else:
                    p_prob = p_prob + np.log(likelihood["pos"]["OOV"])
                if bigram in likelihood["neg"]:
                    n_prob = n_prob + np.log(likelihood["neg"][bigram])
                else:
                    n_prob = n_prob + np.log(likelihood["neg"]["OOV"])
        if p_prob > n_prob:
            hypotheses.append("pos")
        elif p_prob < n_prob:
            hypotheses.append("neg")
        else:
            hypotheses.append("undecided") #rare case n_prob = p_prob
    return hypotheses
    #raise RuntimeError("You need to write this part!")



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors), len(smoothnesses)))
    for m_prior in range(len(priors)):
        sum = len(labels)
        for n_smoothness in range(len(smoothnesses)):
            result = 0
            likelihood = laplace_smoothing(nonstop, smoothnesses[n_smoothness])
            hypotheses = naive_bayes(texts, likelihood, priors[m_prior])
            for label_idx, label in enumerate(labels):
                if label == hypotheses[label_idx]:
                    result += 1
            accuracies[m_prior, n_smoothness] = result/sum
    return accuracies
    #raise RuntimeError("You need to write this part!")
                          