from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np


def dimension_clustering_df(y_true, dim_list, df, one_dim=True, logs=False, logs_freq=1000):
    # The function takes target labels as 0s and 1s and expects 2 clusters only.

    counter = 0
    dim_metrics = []

    for d in dim_list:
        if (counter % logs_freq == 0) and logs:
            print(f'Current dimension: {d}')
        prf_dict = {}
        prf_dict['dim'] = d

        if one_dim:
            X = df.loc[:, d].values.reshape(-1, 1)
        else:
            X = df.loc[:, list(d)].values

        kmeans = KMeans(n_clusters=2, random_state=0, ).fit(X)
        # KMeans returns labels 0 and 1 for clusters, however, we can't be sure if original labels' 1s correspond to
        # KMeans' 1s or 0s. That's why we are going to evaluate 2 cases.
        # Case 1, non-inverted labels of KMeans classifier:
        y_pred1 = kmeans.labels_
        p1_micro, r1_micro, f1_micro, nsamples = precision_recall_fscore_support(y_true, y_pred1, average='micro', zero_division=0)

        # Case 2, inverted labels of KMeans classifier (0s are changed to 1s and 1 are changed to 0s):
        y_pred2 = abs(kmeans.labels_ - 1)
        p2_micro, r2_micro, f2_micro, nsamples = precision_recall_fscore_support(y_true, y_pred2, average='micro', zero_division=0)

        # We can decide to use inverted or non-inverted labels by chosing the class with highest micro average:
        if p1_micro > p2_micro:
            p_macro, r_macro, f_macro, nsamples = precision_recall_fscore_support(y_true, y_pred1, average='macro', zero_division=0)
            prf_dict['P_macro_avg'] = p_macro
            prf_dict['R_macro_avg'] = r_macro
            prf_dict['F_macro_avg'] = f_macro

            p_mixed, r_mixed, f_mixed, nsamples = precision_recall_fscore_support(y_true, y_pred1)
            prf_dict['P_nouns'] = p_mixed[0]
            prf_dict['P_verbs'] = p_mixed[1]
            prf_dict['R_nouns'] = r_mixed[0]
            prf_dict['R_verbs'] = r_mixed[1]
            prf_dict['F_nouns'] = f_mixed[0]
            prf_dict['F_verbs'] = f_mixed[1]

            prf_dict['P_micro_avg'] = p1_micro
            prf_dict['R_micro_avg'] = r1_micro
            prf_dict['F_micro_avg'] = f1_micro


        else:
            p_macro, r_macro, f_macro, nsamples = precision_recall_fscore_support(y_true, y_pred2, average='macro')
            prf_dict['P_macro_avg'] = p_macro
            prf_dict['R_macro_avg'] = r_macro
            prf_dict['F_macro_avg'] = f_macro

            p_mixed, r_mixed, f_mixed, nsamples = precision_recall_fscore_support(y_true, y_pred2)
            prf_dict['P_nouns'] = p_mixed[0]
            prf_dict['P_verbs'] = p_mixed[1]
            prf_dict['R_nouns'] = r_mixed[0]
            prf_dict['R_verbs'] = r_mixed[1]
            prf_dict['F_nouns'] = f_mixed[0]
            prf_dict['F_verbs'] = f_mixed[1]

            prf_dict['P_micro_avg'] = p2_micro
            prf_dict['R_micro_avg'] = r2_micro
            prf_dict['F_micro_avg'] = f2_micro

        dim_metrics.append(prf_dict)
        counter += 1

    dim_metrics_df = pd.DataFrame(dim_metrics)
    dim_metrics_df = dim_metrics_df.set_index('dim')

    return dim_metrics_df.sort_values(by='F_macro_avg', ascending=False)


def add_dimension(original_tuples, str_ind=True, dim_limit=512):
    # By default the indeces of dimension are strings. If str_ind is set to False, indeces can be used as integers.
    # Expects an array of dimensions or an array of dimension tuples.
    # Will return tuples of +1 dimension using the original pairs and their combinations with all dimensions up
    # dimension limit, by default = 512.

    tuple_len = 2 if len(np.array(original_tuples).shape) == 1 else np.array(original_tuples).shape[1] + 1
    tuples = set()

    for tup in original_tuples:
        for i in range(dim_limit):
            if tuple_len > 2:
                tuple_list = list(tup)
            else:
                tuple_list = [tup]
            if str_ind:
                tuple_list.append(str(i))
            else:
                tuple_list.append(i)
            tuple_set = frozenset(tuple_list)
            if len(tuple_set) == tuple_len:
                tuples.add(tuple_set)

    return list(tuples)
