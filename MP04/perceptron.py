# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    W = np.zeros(len(train_set[0]))
    b = 0
    flag = 1
    for i in range(max_iter):
        for j in range(len(train_set)):
            y = train_labels[j]  # label in [-1, 1]
            prediction = np.dot(W, train_set[j, :]) + b
            flag = np.sign(prediction)
            if prediction > 0:
                flag = 1
            else:
                flag = 0
            if flag != y:
                W += (y - flag) * train_set[j, :]
                b += (y - flag)
                flag = False
        if flag:
            break
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    W, b = trainPerceptron(train_set, train_labels, max_iter)
    dev_labels = []
    for i in dev_set:
        prediction = np.dot(i, W) + b
        if prediction >= 0:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
            
    # Train perceptron model and return predicted labels of development set
    return dev_labels
    #return []



