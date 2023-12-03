import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_selection import mutual_info_classif


from scipy.stats import f_oneway


from perceptron import get_perceptron_weights
from preparation import encode_feature, enumerate_and_sort


def get_sequence(dim:list, feature:list):
    return [y[1] for y in sorted(zip(dim, feature), key=lambda x: x[0])]


def aabcc_for_sequence(seq: list):
    # initialisation
    score = 0
    i = 0  # current index
    val = seq[0]  # current value

    while i < len(seq):
        j = 0  # consecutive values counter

        while i < len(seq) and seq[i] == val:
            j += 1
            i += 1

        score += sum([i for i in range(j + 1)])

        if i < len(seq):
            val = seq[i]

    return score

def get_anova_dims(dims_df, feature_vec, pv_threshold=0.01):
    anova_dims = []
    for dim in dims_df.columns:
        sample1 = [x[0] for x in zip(dims_df[dim], feature_vec) if x[1] == 0]
        sample2 = [x[0] for x in zip(dims_df[dim], feature_vec) if x[1] == 1]
        if f_oneway(sample1, sample2).pvalue < pv_threshold:
            anova_dims.append(dim)
    
    return anova_dims


def get_mi_dims(dims_df, feature_vec):
    res = mutual_info_classif(dims_df, feature_vec, discrete_features=[False]*len(dims_df.columns))
    non_indep_dims = [str(x[0]) for x in np.argwhere(res > 0)]
    return non_indep_dims
    


def aabcc(dims_df, feature):
    n_dims = dims_df.shape[1]
    scores = []
    for i in range(n_dims):
        dim = dims_df.iloc[:, i]
        seq = get_sequence(dim, feature)
        score = aabcc_for_sequence(seq)
        scores.append(score)

    return enumerate_and_sort(scores)


def sig_props(dims_df, feature):
    # It is expected that the feature vector contains 2 unique values.
    val1, val2 = feature.unique()

    n_dims = dims_df.shape[1]
    scores = []

    for i in range(n_dims):
        val1_vec = [x[0] for x in zip(dims_df.iloc[:, i], feature) if x[1] == val1]
        val2_vec = [x[0] for x in zip(dims_df.iloc[:, i], feature) if x[1] == val2]
        score = abs(np.mean(val1_vec) - np.mean(val2_vec))
        scores.append(score)

    return enumerate_and_sort(scores)


def correlation(dims_df, feature):
    # It is expected that the feature vector contains 2 unique values.
    # The feature vector is encoded as 0s and 1s.
    feature_encoded = encode_feature(feature)

    n_dims = dims_df.shape[1]
    scores = []

    for i in range(n_dims):
        score = abs(np.corrcoef(dims_df.iloc[:, i], feature_encoded)[0][1])
        scores.append(score)

    return enumerate_and_sort(scores)


def lr(dims_df, feature):
    X_train, X_test, y_train, y_test = train_test_split(dims_df, feature, test_size=0.2, random_state=42)

    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(X_train, y_train)

    return enumerate_and_sort(abs(clf.coef_[0]))


def perceptron(dims_df, feature):
    feature_encoded = encode_feature(feature)

    X_train, X_test, y_train, y_test = train_test_split(dims_df, feature_encoded, test_size=0.2,
                                                        random_state=42)

    weights = None

    for i in range(10):
        if weights is not None:
            weights += get_perceptron_weights(X_train, y_train)
        else:
            weights = get_perceptron_weights(X_train, y_train)

    avg_weights = weights / 10

    return enumerate_and_sort(avg_weights)


def kmeans_1dim(dims_df, feature, start_dim=None, end_dim=None):
    res = []

    feature_encoded = encode_feature(feature)

    if not start_dim:
        start_dim = 0

    if not end_dim:
        end_dim = dims_df.shape[1]

    for i in range(start_dim, end_dim):
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(dims_df.iloc[:, i].values.reshape(-1, 1))

        # Compute the Adjusted Rand Index: the closer to 1, the better
        res.append(adjusted_rand_score(feature_encoded, km.labels_))

    return enumerate_and_sort(res)


def kmeans_multi_dim(dims_df, dim_list, feature):
    feature_encoded = encode_feature(feature)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    try:
        km.fit(dims_df.iloc[:, dim_list].values)
        return adjusted_rand_score(feature_encoded, km.labels_)

    except:
        return None

def run_tests(tests, normalized_dims, feature_vectors, model_labels, report_progress=True):
    all_res = []

    for test in tests:
        if report_progress:
            print(f'Currently running: {test.__name__.upper()}')
        test_res = []
        for i in range(len(model_labels)):
            if report_progress:
                print(f'\t\tModel: {model_labels[i]}')
            test_res.append(test(normalized_dims[i], feature_vectors[i]))
        all_res.append(test_res)
        if report_progress:
            print('Done\n=======================================')

    return all_res


def report(all_res, tests, model_labels):
    all_res_df = pd.DataFrame(columns=['Min', 'Max', 'Mean', '25%', '50%', '75%', '95%', '97.7%', '99.9%'])
    for i in range(len(tests)):
        test_df = score_comparison(all_res[i], model_labels, tests[i].__name__)
        all_res_df = pd.concat([all_res_df, test_df])

    return all_res_df


def dimensions_by_percentile(results, model_labels, q):
    dimensions = {}
    n = len(results)

    for i in range(n):
        res = results[i]
        scores = np.array([x[1] for x in res])
        percentile = np.percentile(scores, q)
        dimensions[model_labels[i]] = np.array([x[0] for x in res if x[1] >= percentile])

    return dimensions


def dimensions_report(all_res, tests, model_labels, q):
    report_df = pd.DataFrame(columns=model_labels)
    for i in range(len(tests)):
        test_dimensions = dimensions_by_percentile(all_res[i], model_labels, q)
        report_df.loc[tests[i].__name__.upper()] = test_dimensions

    return report_df


def repeated_dimensions(dimensions_df, model_labels):
    repeated_dimensions_df = pd.DataFrame(columns=[ '1 test', '2 tests',
                                                   '3 tests', '4 tests', '5 tests', '6 tests'])

    for i in range(len(model_labels)):
        dims, counts = np.unique(np.concatenate(dimensions_df[model_labels[i]], axis=0), return_counts=True)
        repeated_dimensions_df.loc[model_labels[i]] = {
            '1 test':  [x[0] for x in zip(dims, counts) if x[1] >= 1],
            '2 tests': [x[0] for x in zip(dims, counts) if x[1] >= 2],
            '3 tests': [x[0] for x in zip(dims, counts) if x[1] >= 3],
            '4 tests': [x[0] for x in zip(dims, counts) if x[1] >= 4],
            '5 tests': [x[0] for x in zip(dims, counts) if x[1] >= 5],
            '6 tests': [x[0] for x in zip(dims, counts) if x[1] >= 6],
        }

    return repeated_dimensions_df


def score_comparison(results, model_labels, test_name):
    res_df = pd.DataFrame(columns=['Min', 'Max', 'Mean', '25%', '50%', '75%', '95%', '97.7%', '99.9%'])

    n = len(results)

    for i in range(n):
        res = results[i]
        scores = np.array([x[1] for x in res])
        describe_dict = {
            'Min': scores.min(),
            'Max': scores.max(),
            'Mean': scores.mean(),
            '25%': np.percentile(scores, 25),
            '50%': np.percentile(scores, 50),
            '75%': np.percentile(scores, 75),
            '95%': np.percentile(scores, 95),
            '97.7%': np.percentile(scores, 97.7),
            '99.9%': np.percentile(scores, 99.9)
        }

        res_df.loc[f'{test_name.upper()}_{model_labels[i]}'] = describe_dict

    return res_df


