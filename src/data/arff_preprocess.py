import numpy as np
import pandas as pd
import sys
from scipy.io import arff

def encode_category(data):
    data_ = np.zeros(data.shape)
    uniques = []
    for i in range(data.shape[-1]):
        unique_labels = np.unique(data[:, i])
        uniques.append(len(unique_labels))
        d = dict((x, i) for i, x in enumerate(unique_labels))
        data_[:, i] = np.array([d[x] for x in data[:, i]])
    return data_, tuple(uniques)

def arff_preproc(filename: str, label_name='class'):
    data, meta = arff.loadarff(f"data/raw/{filename}.arff")
    data = np.array(data.tolist())  # unless its one dim vector with np.void
    names = np.array(meta.names())
    types = np.array(meta.types())

    categ_features = (types == 'nominal') & (names != label_name)
    cont_features = (types == 'numeric')

    x_categ = data[:, categ_features].astype('<U14')
    x_cont = data[:, cont_features].astype(np.float64)
    labels = data[:, names == label_name].astype('<U14')    # if classification task

    # закодируем фичи, уберем постоянные значения в непрерывных фичах
    x_categ_, _ = encode_category(x_categ)
    labels, _ = encode_category(labels)
    x_cont = x_cont[:, np.std(x_cont, axis=0) != 0]

    pd.DataFrame(data=x_categ_).to_csv(f'data/preprocessed/{filename}/categ.csv', index=False)    # header, index ?
    pd.DataFrame(data=x_cont).to_csv(f'data/preprocessed/{filename}/cont.csv', index=False)
    pd.DataFrame(data=labels).to_csv(f'data/preprocessed/{filename}/labels.csv', index=False)

if __name__ == '__main__':
    filename = sys.argv[1]
    arff_preproc(filename)



