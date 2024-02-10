import time

import numpy as np

from train.train import model_training
from accuracy import accuracy
from model.hmm import HMM
from data.data import Dataset


def sentence_tagging(test_data, model, tags):

    tagging = []

    for line in test_data:
        print("line: ", line.words)
        for word in line.words:
            print("word: ", word)
            if word not in model.obs_dict:
                model.obs_dict[word] = len(model.obs_dict)
                col = np.ones((len(model.pi), 1)) * 1e-6
                model.B = np.hstack((model.B, col))
        tagging.append(model.viterbi(line.words))
    return tagging

def speech_tagging_test():
    st_time = time.time()
    data = Dataset("pos_tags.txt", "pos_sentences.txt", train_test_split=0.8, seed=0)

    data.train_data = data.train_data[:100]

    data.test_data = data.test_data[:10]

    model = model_training(data.train_data, data.tags)

    tagging = sentence_tagging(data.test_data, model, data.tags)

    total_words = 0
    total_correct = 0
    for i in range(len(tagging)):

        correct, words, accur = accuracy(tagging[i], data.test_data[i].tags)
        total_words += words
        total_correct += correct

    print("Your total accuracy for tagging: ", total_correct*1.0/total_words)
    print("Expected total accuracy is about: ", 0.85)

    en_time = time.time()
    print("Sentence tagging total time: ", en_time - st_time)


def custom_test(words):

    data = Dataset("pos_tags.txt", "pos_sentences.txt", train_test_split=0.8, seed=0)
    data.train_data = data.train_data[:100]
    model = model_training(data.train_data, data.tags)

    tagging = []
    for word in words:
        if word not in model.obs_dict:
            model.obs_dict[word] = len(model.obs_dict)
            col = np.ones((len(model.pi), 1)) * 1e-6
            model.B = np.hstack((model.B, col))

    tagging.append(model.viterbi(words))

    return tagging


if __name__ == "__main__":

    input = str(input("Enter a sentence: "))
    words = input.split(" ")
    print(custom_test(words))