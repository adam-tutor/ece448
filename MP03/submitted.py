'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here
epsilon = 1e-5

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tracker_tag = {}
    tagset_output = []
    counter_tag = Counter()
    for sentence in train:
        for word, tag in sentence:
                counter_tag[tag] += 1
                if word not in tracker_tag:
                        tracker_tag[word] = Counter()
                tracker_tag[word][tag] += 1

    for sentence in test:
        sentence_list = []
        for word in sentence:
                if word in tracker_tag:
                        sentence_list.append((word, tracker_tag[word].most_common(1)[0][0]))
                else:
                        sentence_list.append((word, counter_tag.most_common(1)[0][0]))
        tagset_output.append(sentence_list)
    return tagset_output
  
    #raise NotImplementedError("You need to write this part!")

def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #Step 1: Count occurrences of tags, tag pairs, tag/word pairs.
    
    tag_count = Counter()
    pair_count = defaultdict(Counter)
    tracker_tag = defaultdict(Counter)
    tag_word_count = defaultdict(Counter)
    word_count = defaultdict(int)

    for sentence in train:
        prev_tag = 'START'
        for word, tag in sentence:
            tag_count[tag] += 1
            pair_count[prev_tag][tag] += 1
            tracker_tag[word][tag] += 1
            tag_word_count[tag][word] += 1
            word_count[word] += 1
            prev_tag = tag
    
    #Step 2: Compute smoothed probabilities
    #Step 2.1: Initial prob
    prob_initial = defaultdict(float)
    for tag, key in pair_count['START'].items():
        prob_initial[tag] = np.log(key + epsilon) - np.log(epsilon * (len(tag_count)) + tag_count['START'])

    #Step 2.2: Transition prob
    prob_transition = defaultdict(lambda: defaultdict(float))
    for tag in tag_count.keys():
        for tag_iter in tag_count:
            prob_transition[tag][tag_iter] = np.log(pair_count[tag][tag_iter] + epsilon) - np.log(sum(pair_count[tag].values()) + (len(tag_count)) * epsilon)
    
    #Step 2.3: Emission prob
    prob_emission = defaultdict(lambda: defaultdict(float))
    for tag in tag_count.keys():
        prob_emission['UNKNOWN'][tag] = np.log(epsilon) - np.log(sum(tag_word_count[tag].values()) + epsilon * len(word_count))
        for word in tag_word_count[tag]:
            if tag in tracker_tag[word]:
                prob_emission[word][tag] = np.log(tracker_tag[word][tag] + epsilon) - np.log(sum(tag_word_count[tag].values()) + epsilon * len(word_count))
    #Step 4: Construct the trellis.
    best_path = []
    for sentence in test:
        trellis = defaultdict(lambda:defaultdict(float))
        parent = defaultdict(lambda:defaultdict(float)) 
        sentence_idx = len(sentence)-1
        if sentence[0] not in prob_emission:
                sentence[0] = 'UNKNOWN' #initialize probabilities starting at sentence[0]
        for tag, pb in prob_emission[sentence[0]].items():
                trellis[0][tag] = pb + prob_initial[tag]
                
        for i in range(1, len(sentence)):
                word = sentence[i]
                if word not in prob_emission:
                        word = 'UNKNOWN'
                for tag in prob_emission[word]:
                        max_prob = -float('inf')
                        for prev_tag in trellis[i-1]:
                                if max_prob < (trellis[i-1][prev_tag] + prob_transition[prev_tag][tag] + prob_emission[word][tag]):
                                        max_prob = trellis[i-1][prev_tag] + prob_transition[prev_tag][tag] + prob_emission[word][tag] #update new maximum probability
                                        max_tag = prev_tag
                        trellis[i][tag] = max_prob
                        parent[i][tag] = max_tag
        trellis_tags = []
        curr_tag = "END"
        for i in range(sentence_idx, -1, -1):
                trellis_tags.append((sentence[i], curr_tag)) #retrieve final prediction from stored probs
                if i > 0:
                        curr_tag = parent[i][curr_tag]
        trellis_tags.reverse()
        best_path.append(trellis_tags)

    #Step 5: Return best path thru trellis
    return best_path
    #raise NotImplementedError("You need to write this part!")


def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



