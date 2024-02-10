from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        for s in range(0, S):
            alpha[s, 0] = self.pi[s] * self.B[s, O[0]]
        for t in range(1, L):
            for s in range(0, S):
                alpha[s, t] = self.B[s, O[t]] * np.sum(self.A[:,s] * alpha[:, t -1])
        return alpha

    def backward(self, Osequence):
        
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        for s in range(0, S):
            beta[s, L - 1] = 1
        for t in range(L - 2, -1, -1):
            for s in range(0, S):
                beta[s, t] = np.sum(self.A[s] * self.B[:, O[t + 1]] * beta[:, t + 1])
        return beta
    
    def sequence_prob(self, Osequence):
        alpha = self.forward(Osequence)
        return np.sum(alpha[:, len(Osequence) - 1])

    def posterior_prob(self, Osequence):

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        gamma = (alpha * beta) / np.sum(alpha[:, len(Osequence) - 1])
        return gamma
    
    def likelihood_prob(self, Osequence):

        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        O = self.find_item(Osequence)

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        denominator = np.sum(alpha[:, len(Osequence) - 1]);
        for s in range(0, S):
            for s1 in range(0, S):
                for t in range(0, L - 1):
                    prob[s][s1][t] = (alpha[s,t] * self.A[s, s1] * self.B[s1, O[t + 1]] * beta[s1, t + 1]) / denominator
        return prob

    def viterbi(self, Osequence):

        path = []
        S = len(self.pi) 
        L = len(Osequence) 

        delta = np.zeros([S, L])
        D = np.zeros([S, L])

        O = self.find_item(Osequence)
        for s in range(0, S):
            delta[s, 0] = self.pi[s] * self.B[s, O[0]]
        for t in range(1, L):
            for s in range(0, S):
                delta[s, t] = self.B[s, O[t]] * np.max(self.A[:,s] * delta[:, t -1])
                D[s,t] = np.argmax(self.A[:,s] * delta[:, t - 1])
        prev = np.argmax(delta[:, L - 1])
        path.append(self.find_key(self.state_dict, prev))
        for t in range(L - 1, 0, -1):
            path.append(self.find_key(self.state_dict, D[prev, t]))
            prev = int(D[prev, t])
        path.reverse()            
        return path


    def find_key(self, obs_dict, idx):

        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):

        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
