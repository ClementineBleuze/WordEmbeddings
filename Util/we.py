import torch
from transformers import FlaubertModel, FlaubertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np
import pandas as pd

import seaborn
import matplotlib.pyplot as plt


class WENotFound(Exception):
    pass


def initiate_model(name='flaubert/flaubert_small_cased'):
    tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_small_cased")
    model = AutoModelForMaskedLM.from_pretrained("flaubert/flaubert_small_cased")

    #flaubert, log = FlaubertModel.from_pretrained(name, output_loading_info=True)
    #flaubert_tokenizer = FlaubertTokenizer.from_pretrained(name, do_lowercase=False)
    return model, tokenizer


def get_we(model, tokenizer, word):
    # The tokenizer creates a vector of the following format [0, <word_id>, 1] where 0 is the code for [CLS]
    # and 1 is the code for the string end. That's why we're only taking the second element of array for getting its
    # embedding. If the word doesn't exist in the vocabulary, the tokenizer would attempt to it by parts like [0, <
    # id of the first part>, <id of the second part>, ..., 1]. That's why we assume that the word exists in the
    # vocabulary only if the encoding is of length 3.

    encoding = tokenizer.encode(word)
    if len(encoding) != 3:
        raise WENotFound(f'{word}: the word doesn\'t exist in the vocab')

    word_id = encoding[1]

    # We create a Tensor since the model works with this format
    token_ids = torch.tensor([[word_id]])

    # Get the last layer of the model with the final embedding. `last_layer` is a Tensor
    last_layer = model(token_ids)[0]

    # Convert the Tensor to a numpy array. Since Tensor is 3-Dimensional, we need to flatten the result as well
    we = last_layer.detach().numpy().flatten()

    return we


def create_words_df(model, tokenizer, words, transposed=True, progress=False, progress_step=1000):
    words_we = {}
    counter = 0
    for w in words:
        if (counter % progress_step == 0) and progress:
            print('.', end='')
        try:
            words_we[w] = get_we(model, tokenizer, w)
        except WENotFound:
            pass
        counter += 1

    words_df = pd.DataFrame(words_we)
    return words_df.transpose() if transposed else words_df


def plot_we_heatmap(we_df, label='', size=(20, 3), cmap='YlGnBu'):
    n_dim = we_df.shape[1]
    i = 0
    while i < n_dim:
        j = min(i + 100, n_dim)
        fig, ax = plt.subplots(figsize=size)
        seaborn.heatmap(we_df.iloc[:, i:j], cmap=cmap)
        ax.set_title(f'{label}. Dim {i}-{j-1}')
        plt.show()
        i = j