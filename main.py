import numpy as np

from data.data import Dataset
from train.train import model_training



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
    print(custom_test(words)[0])