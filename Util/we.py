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
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name)

    # flaubert, log = FlaubertModel.from_pretrained(name, output_loading_info=True)
    # flaubert_tokenizer = FlaubertTokenizer.from_pretrained(name, do_lowercase=False)
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

    # Get hidden states of the last layer. The result is a Tensor.
    last_layer = model(token_ids, output_hidden_states=True).hidden_states[-1]

    # Convert the Tensor to a numpy array. Since Tensor is 3-Dimensional, we need to flatten the result as well
    we = last_layer.detach().numpy().flatten()

    return we


def create_we_df(model, tokenizer, words, progress=False, progress_step=1000):
    # 'model' and 'tokenizer' are expected to be initialized via 'iniate_model' function
    # 'words' expected to be a Pandas DataFrame with wordforms as its index and features as columns
    # If progress == true, the function will output a dot for every progress step.
    # As a result a Pandas DataFrame is returned where index column is the word form, integer-titled columns are
    # corresponding dimensions and string-titled columns are grammatical features (that were given in columns) of the
    # 'words' DataFrame.

    words_list = []
    features = words.columns

    counter = 0
    for w in words.index:
        if (counter % progress_step == 0) and progress:
            print('.', end='')
        try:
            we = get_we(model, tokenizer, w)
            word_dict = {x[0]: x[1] for x in enumerate(we)}
            word_dict['Word'] = w
            for f in features:
                word_dict[f] = words.loc[w][f]
            words_list.append(word_dict)
        except WENotFound:
            pass
        counter += 1

    words_df = pd.DataFrame(words_list)
    words_df = words_df.set_index('Word')
    return words_df


def plot_we_heatmap(we_df, label='', size=(20, 3), cmap='YlGnBu', save=None):
    n_dim = we_df.shape[1]
    i = 0
    while i < n_dim:
        j = min(i + 103, n_dim)
        fig, ax = plt.subplots(figsize=size)
        seaborn.heatmap(we_df.iloc[:, i:j], cmap=cmap)
        ax.set_title(f'{label}. Dim {i}-{j - 1}')
        if save:
            plt.savefig(f"{save}_{i}-{j - 1}.png", bbox_inches='tight')
        plt.show()
        i = j
