import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from tab_transformer import *


def my_train_test_split(X1, X2, y, test_size=0.2, val_size=0.05, random_state=42):
    # X1 - categorical, X2 - continious, y - labels
    assert X1.shape[0] == X2.shape[0] == y.shape[0]
    n1 = int(np.around(test_size * y.shape[0]))
    n2 = int(np.around((val_size + test_size) * y.shape[0]))
    if random_state is not None:
        np.random.seed(random_state)

    idx = np.random.permutation(y.shape[0])

    X1_test = X1[idx][:n1]
    X2_test = X2[idx][:n1]
    y_test = y[idx][:n1]

    X1_val = X1[idx][n1:n2]
    X2_val = X2[idx][n1:n2]
    y_val = y[idx][n1:n2]

    X1_train = X1[idx][n2:]
    X2_train = X2[idx][n2:]
    y_train = y[idx][n2:]

    return X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test


class MyDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def count_acc(y_true, y_pred):
    # x = (torch.round(nn.Sigmoid()(y_pred)) == y_true).sum().float().item() / y_true.shape[0] / y_true.shape[1]
    # y = accuracy_score(torch.round(nn.Sigmoid()(y_pred)).detach().numpy(), y_true.detach().numpy())
    # assert x == y
    return (torch.round(nn.Sigmoid()(y_pred)) == y_true).sum().float().item() / y_true.shape[0]
    # return accuracy_score(torch.round(nn.Sigmoid()(y_pred)).detach().numpy(), y_true.detach().numpy())
    # return x


def count_auc(model, device, *dataloaders):
    y_true, y_pred = [], []
    with torch.no_grad():
        for dataloader in dataloaders:
            for i, (x_categ, x_cont, labels) in enumerate(dataloader):
                x_categ, x_cont, labels = x_categ.long().to(device), x_cont.float().to(device), labels.float().to(
                    device)
                pred = nn.Sigmoid()(model.forward(x_categ, x_cont))
                y_true.extend(list(labels.detach().cpu().numpy()))
                y_pred.extend(list(pred.detach().cpu().numpy()))

    return roc_auc_score(y_true, y_pred)


def count_pres_rec_f1(model, device, *dataloaders):
    y_true, y_pred = [], []
    with torch.no_grad():
        for dataloader in dataloaders:
            for i, (x_categ, x_cont, labels) in enumerate(dataloader):
                x_categ, x_cont, labels = x_categ.long().to(device), x_cont.float().to(device), labels.float().to(
                    device)
                pred = nn.Sigmoid()(model.forward(x_categ, x_cont))
                y_true.extend(list(labels.detach().cpu().numpy()))
                y_pred.extend(list(torch.round(pred).detach().cpu().numpy()))

    return precision_score(y_true, y_pred, zero_division=0), \
           recall_score(y_true, y_pred, zero_division=0), \
           f1_score(y_true, y_pred, zero_division=0)


def single_pass(model, dataloader, loss_func, device, optim=None):
    loss_count, acc_count = 0, 0
    for i, (x_categ, x_cont, labels) in enumerate(dataloader):
        x_categ, x_cont, labels = x_categ.long().to(device), x_cont.float().to(device), labels.float().to(device)
        pred = model.forward(x_categ, x_cont)
        loss = loss_func(pred, labels)
        loss_count += loss.item()
        acc_count += count_acc(labels, pred)
        # roc_auc += roc_auc_score(labels, pred, average='macro')
        if optim is not None:
            loss.backward()
            optim.step()
    return loss_count / len(dataloader), acc_count / len(dataloader)


def plot_results(ax, train_results: list, val_results: list, test_result, label):
    epochs = np.arange(1, len(train_results) + 1)
    ax.plot(epochs, train_results, label='train')
    ax.plot(epochs, val_results, label='validation')
    ax.plot(epochs[-1], test_result,
            marker='o', linestyle='none', label='test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(label)
    ax.grid(linestyle=':')
    ax.legend()


def train_model(model, loss, optim, epochs, device, dataloaders, single_pass=single_pass):
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    train_loss_all, val_loss_all, train_acc_all, val_acc_all = [], [], [], []
    # training loop
    for epoch in range(epochs):
        # train
        # print('train')
        train_loss, train_acc = single_pass(model, dataloader_train, loss, device, optim)
        # print('val')
        # validation
        with torch.no_grad():
            val_loss, val_acc = single_pass(model, dataloader_val, loss, device)

        print(
            f'epoch {epoch}, train_loss={train_loss}, validation_loss={val_loss}, train_acc={train_acc}, val_acc={val_acc}')

        train_loss_all.append(train_loss)
        val_loss_all.append(val_loss)
        train_acc_all.append(train_acc)
        val_acc_all.append(val_acc)

    # test
    model.eval()
    with torch.no_grad():
        test_loss, test_acc = single_pass(model, dataloader_test, loss, device)
        test_AUC = count_auc(model, device, dataloader_test)
        pres, rec, f1 = count_pres_rec_f1(model, device, dataloader_test)

        print(f'test_loss={test_loss}, test_acc={test_acc}')
        print('test_AUC=', test_AUC)
        print('pres=', pres, 'rec=', rec, 'f1=', f1)
    model.train()
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, test_loss, test_acc


def tab_train(filename, mode='ordinary', seed=42):  # mlm_single / mlm_different
    if mode == 'ordinary':
        path = os.path.join('data/nan_as_categ', filename)
    elif mode == 'naive':
        path = os.path.join('data/recovered', f'{filename}_naive')
    elif mode == 'mlm_single':
        path = os.path.join('data/recovered', f'{filename}_mlm_single')
    elif mode == 'mlm_different':
        path = os.path.join('data/recovered', f'{filename}_mlm_different')

    data_categ = pd.read_csv(os.path.join(path, 'categ.csv')).to_numpy()
    data_cont = pd.read_csv(os.path.join(path, 'cont.csv')).to_numpy()
    data_labels = pd.read_csv(os.path.join(path, 'labels.csv')).to_numpy()
    # print(data_cont.shape)

    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = my_train_test_split(data_categ,
                                                                                                       data_cont,
                                                                                                       data_labels,
                                                                                                       test_size=0.2,
                                                                                                       val_size=0.05,
                                                                                                       random_state=seed)

    cont_mean_std = np.array([X2_train.mean(axis=0), X2_train.std(axis=0)]).transpose(1, 0)
    cont_mean_std = torch.Tensor(cont_mean_std)

    categories = tuple(len(np.unique(data_categ[:, i])) for i in range(data_categ.shape[1]))
    print(categories)
    model = TabTransformer(
        categories=categories,  # tuple containing the number of unique values within each category
        num_continuous=data_cont.shape[-1],  # number of continuous values
        dim=32,  # dimension, paper set at 32
        dim_out=1,  # binary prediction, but could be anything
        depth=6,  # depth, paper recommended 6
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=None,  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        continuous_mean_std=cont_mean_std,  # (optional) - normalize the continuous values before layer norm
        seed=seed
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 1900
    epochs = 10
    lr = 1e-4
    optim = Adam(model.parameters(), lr=lr)
    loss = F.binary_cross_entropy_with_logits

    dataset_train = MyDataset(X1_train, X2_train, y_train)
    dataset_val = MyDataset(X1_val, X2_val, y_val)
    dataset_test = MyDataset(X1_test, X2_test, y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=64)
    dataloader_test = DataLoader(dataset_test, batch_size=64)
    dataloaders = [dataloader_train, dataloader_val, dataloader_test]

    train_loss_all, val_loss_all, train_acc_all, val_acc_all, test_loss, test_acc = \
        train_model(model, loss, optim, epochs, device, dataloaders)

    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, test_loss, test_acc


def my_subplots(train_loss, val_loss, train_acc, val_acc, test_loss, test_acc):
    _, ax1 = plt.subplots()
    plot_results(ax1, train_loss, val_loss, test_loss, 'Loss')
    _, ax2 = plt.subplots()
    plot_results(ax2, train_acc, val_acc, test_acc, 'Accuracy')
    plt.show()


def main():
    filename = sys.argv[1]
    my_subplots(*tab_train(filename))


if __name__ == '__main__':
    main()
