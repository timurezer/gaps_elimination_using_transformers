import numpy as np
import pandas as pd
import sys
import os
from arff_preprocess import encode_category

def add_Nans(filename, power=0.15, feature='all', seed=42):
    np.random.seed(seed)
    path = 'data/preprocessed'
    categ = pd.read_csv(os.path.join(path, filename, 'categ.csv')).to_numpy().astype(float)
    cont = pd.read_csv(os.path.join(path, filename, 'cont.csv')).to_numpy().astype(float)
    labels = pd.read_csv(os.path.join(path, filename, 'labels.csv')).to_numpy().astype(float)
    true_labes = []
    idx = np.random.choice(categ.shape[0], int(power * categ.shape[0]), replace=False)
    if isinstance(feature, int):
        # удалаем только в выбранном столбце feature
        assert feature < categ.shape[1]
        true_labes = categ[idx, feature]
        categ[idx, feature] = -1    # np.nan
    else:
        if feature == 'all':
            # удаляет во всех столбцах таблицы
            cols = np.random.choice(categ.shape[1], len(idx)).astype(int)
        elif isinstance(feature, list):
            # удаляет только в столбцах feature
            assert all(np.array(feature) < categ.shape[1])
            cols = np.array(feature)[np.random.choice(len(feature), len(idx)).astype(int)]
        true_labes = categ[idx, cols]
        categ[idx, cols] = -1    # np.nan

    categ, _ = encode_category(categ)
    
    pd.DataFrame(data=categ).to_csv(f'data/nan_as_categ/{filename}/categ.csv', index=False)
    pd.DataFrame(data=cont).to_csv(f'data/nan_as_categ/{filename}/cont.csv', index=False)
    pd.DataFrame(data=labels).to_csv(f'data/nan_as_categ/{filename}/labels.csv', index=False)
    # pd.DataFrame(data=true_labes).to_csv(f'data/with_nans/{filename}/true_labels.csv', index=False)

def main(seed=42):
    filename = sys.argv[1]
    power = float(sys.argv[2])
    print(filename)
    add_Nans(filename, power=power, feature=[1, 3], seed=seed)

if __name__ == '__main__':
    main()
