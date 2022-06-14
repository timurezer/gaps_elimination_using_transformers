import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier


def main(filename, mode='ordinary', seed=42, nan_mode='Min', eval='val'):
    path = ''
    if mode == 'ordinary':
        path = os.path.join('/content/drive/MyDrive/HSE/NIR/data/nan_as_categ', filename)
    elif mode == 'naive':
        path = os.path.join('/content/drive/MyDrive/HSE/NIR/data/recovered', f'{filename}_naive')
    elif mode == 'mlm_single':
        path = os.path.join('/content/drive/MyDrive/HSE/NIR/data/recovered', f'{filename}_mlm_single')
    elif mode == 'mlm_different':
        path = os.path.join('/content/drive/MyDrive/HSE/NIR/data/recovered', f'{filename}_mlm_different')

    data_categ = pd.read_csv(os.path.join(path, 'categ.csv')).to_numpy()
    data_cont = pd.read_csv(os.path.join(path, 'cont.csv')).to_numpy()
    data = np.hstack((data_categ, data_cont))

    labels = pd.read_csv(os.path.join(path, 'labels.csv')).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.8, random_state=42)

    clf = CatBoostClassifier(
        custom_loss=['Accuracy'],
        random_seed=seed,
        logging_level='Silent',
        loss_function='Logloss',  # NLLL
        nan_mode=nan_mode,
        l2_leaf_reg=0.01,
        task_type="GPU"
    )
    clf.fit(X_train, y_train)

    if eval == 'val':
        y_val_pred = clf.predict(X_val)
        print('acc:', accuracy_score(y_val, y_val_pred), 'pres:', precision_score(y_val, y_val_pred), 'rec:',
              recall_score(y_val, y_val_pred))
        print('AUC', roc_auc_score(y_val, y_val_pred))

    else:
        y_test_pred = clf.predict(X_test)
        print(seed)
        acc = accuracy_score(y_test, y_test_pred)
        pres = precision_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_pred)
        print('acc:', acc, 'pres:', pres, 'rec:', rec, 'AUC', auc)

        return acc, auc


dataset = 'adult'

seeds = [42, 10, 100, 1000, 10000]

"""# MLM_SINGLE"""

acc_all, auc_all = [], []
mode = 'mlm_single'

for seed in seeds:
    acc, auc = main(dataset, mode=mode, seed=seed, eval='test')
    acc_all.append(acc)
    auc_all.append(auc)

print('acc', np.mean(acc_all), np.std(acc_all))
print('auc', np.mean(auc_all), np.std(auc_all))

"""# MLM_DIFFERENT"""

acc_all, auc_all = [], []
mode = 'mlm_different'

for seed in seeds:
    acc, auc = main(dataset, mode=mode, seed=seed, eval='test')
    acc_all.append(acc)
    auc_all.append(auc)

print('acc', np.mean(acc_all), np.std(acc_all))
print('auc', np.mean(auc_all), np.std(auc_all))

"""# NAIVE"""

acc_all, auc_all = [], []
mode = 'naive'

for seed in seeds:
    acc, auc = main(dataset, mode=mode, seed=seed, eval='test')
    acc_all.append(acc)
    auc_all.append(auc)

print('acc', np.mean(acc_all), np.std(acc_all))
print('auc', np.mean(auc_all), np.std(auc_all))

"""# ORDINARY"""

acc_all, auc_all = [], []
mode = 'ordinary'

for seed in seeds:
    acc, auc = main(dataset, mode=mode, seed=seed, eval='test')
    acc_all.append(acc)
    auc_all.append(auc)

print('acc', np.mean(acc_all), np.std(acc_all))
print('auc', np.mean(auc_all), np.std(auc_all))
