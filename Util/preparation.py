import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_datasets(path, model_labels, file_name):
    # Returns a list of DataFrames for each model for a selected file.
    # It is assumed that model names are folder names and files to read are named the same way for every model.
    # Path is to the root folder containing model folder inside. Path shouldn't end with a slash.

    return [pd.read_csv(f'{path}/{m_label}/{file_name}', index_col=0, low_memory=False) for m_label in model_labels]


def enumerate_and_sort(array, reverse=True):
    return sorted(enumerate(array), key=lambda x: x[1], reverse=reverse)


def encode_feature(feature):
    le = LabelEncoder()
    le.fit(feature.unique())
    feature_encoded = le.transform(feature)
    return feature_encoded


def prepare_dataset(dataset, feature_col_count, feature_name, normalize=False, encode=False, encode_as1=None, split=False, balance=False):
    # The function assumes that `dataset` is a Pandas DataFrame where last n columns are feature columns
    # and all [0:n] columns are dimensions of the WE.
    # n is equal to feature_col_count.
    # The function will return normalized values of dimensions of WE and the feature vector.
    # The words are shuffled prior to the return.

    
    # Encode the feauture using LabelEncoder
    if encode:
        if encode_as1:
            dataset[feature_name] = (dataset[feature_name] == encode_as1).apply(int)
        else:
            dataset[feature_name] = encode_feature(dataset[feature_name])
        
    if balance:
        new_dataset = []
        min_values = min([len(dataset[dataset[feature_name] == x]) for x in set(dataset[feature_name])])

        for feature_value in set(dataset[feature_name]):
            new_dataset.append(dataset[dataset[feature_name] == feature_value].sample(n=min_values))
        
        dataset = pd.concat(new_dataset)
    
    dims = dataset.iloc[:, :-feature_col_count]
    
    # Normalize dimensions using min and max
    if normalize:
        dims = (dims - dims.min())/(dims.max() - dims.min())
    
    X, y = shuffle(dims, dataset[feature_name])
    
    # Split into test and train if specified. The split is 80/20
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y

