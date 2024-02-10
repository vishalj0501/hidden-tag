import numpy as np

from model.hmm import HMM

def get_unique_words(data):
    unique_words = {}
    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words



def model_training(train_data, tags):


    unique_words = get_unique_words(train_data)

    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    
    index = 0
    for tag in tags:
        tag2idx[tag] = index
        index = index + 1

    index = 0
    for word in unique_words:
        word2idx[word] = index;
        index = index + 1

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))

    for line in train_data:

        pi[tag2idx[line.tags[0]]] += 1
        B[tag2idx[line.tags[0]], word2idx[line.words[0]]] += 1
        
        for i in range(1, line.length):
            A[tag2idx[line.tags[i - 1]], tag2idx[line.tags[i]]] += 1
            B[tag2idx[line.tags[i]], word2idx[line.words[i]]] += 1
            
    pi = pi / np.sum(pi)
    
    for i in range(0, S):
        A[i, :] = A[i, :] / np.sum(A[i, :])
        B[i, :] = B[i, :] / np.sum(B[i, :])

    model = HMM(pi, A, B, word2idx, tag2idx)

    return model

