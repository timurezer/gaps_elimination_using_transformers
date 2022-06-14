from train_simple_tab import *

def train(model, data_categ, data_cont, data_labels, seed=42):

    X1_train, X2_train, y_train, X1_val, X2_val, y_val, X1_test, X2_test, y_test = my_train_test_split(data_categ,
                                                                                                       data_cont,
                                                                                                       data_labels,
                                                                                                       test_size=0.2,
                                                                                                       val_size=0.05,
                                                                                                       random_state=seed)

    continuous_mean_std = np.array([X2_train.mean(axis=0), X2_train.std(axis=0)]).transpose(1, 0)
    continuous_mean_std = torch.Tensor(continuous_mean_std)
    model.register_buffer('continuous_mean_std', continuous_mean_std)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 1900
    epochs = 10  # 10
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

    train_loss_all, val_loss_all, train_acc_all, val_acc_all, test_loss, test_acc =\
        train_model(model, loss, optim, epochs, device, dataloaders)
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, test_loss, test_acc


def separate_nans(data_categ, data_cont, data_labels):
    # find rows with NaNs and separate them from pure data
    # Nans rows are used to recover data in the missings
    idx = np.array([i for i, x in enumerate(data_categ) if not any(np.isnan(x))])
    idx = np.in1d(np.arange(data_categ.shape[0]), idx)    # idx for clear rows
    categ_clear = data_categ[idx]
    cont_clear = data_cont[idx]
    labels_clear = data_labels[idx]

    nidx = np.logical_not(idx)
    categ_nan = data_categ[nidx]
    cont_nan = data_cont[nidx]
    labels_nan = data_labels[np.logical_not(idx)]
    nans_pos = np.where(np.isnan(categ_nan))

    # we suppose that we have only one Nan in rows
    return categ_clear, categ_nan, cont_clear, cont_nan, labels_clear, labels_nan, nans_pos


def main(seed=42):
    filename = sys.argv[1]
    path = os.path.join('data/preprocessed', filename)
    path_nans = os.path.join('data/with_nans', filename)
    data_categ_all = pd.read_csv(os.path.join(path_nans, 'categ.csv')).to_numpy()    # read with nans file
    true_labels = pd.read_csv(os.path.join(path_nans, 'true_labels.csv')).to_numpy().reshape(-1)
    data_cont = pd.read_csv(os.path.join(path, 'cont.csv')).to_numpy()
    data_labels = pd.read_csv(os.path.join(path, 'labels.csv')).to_numpy()

    categ_clear, categ_nan, cont_clear, cont_nan, labels_clear, labels_nan, nans_pos =\
        separate_nans(data_categ_all, data_cont, data_labels)    # data_categ is pure of NaNs

    categories = tuple(len(np.unique(categ_clear[:, i])) for i in range(categ_clear.shape[1]))

    model = TabTransformer(
        categories=categories,  # tuple containing the number of unique values within each category
        num_continuous=data_cont.shape[-1],  # number of continuous values
        dim=32,  # dimension, paper set at 32
        dim_out=1,  # binary prediction, but could be anything
        depth=6,  # depth, paper recommended 6
        heads=2,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=None,  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        continuous_mean_std=None,  # (optional) - normalize the continuous values before layer norm
        seed=seed
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(model, categ_clear, cont_clear, labels_clear)
    # recover labels
    recovered_labels = model.naive_recover(categ_nan, nans_pos[1], device)
    print('acc', accuracy_score(true_labels, recovered_labels))

    # insert them into dataset
    categ_nan[nans_pos] = recovered_labels
    new_categ = np.vstack((categ_clear, categ_nan))
    new_cont = np.vstack((cont_clear, cont_nan))
    new_labels = np.vstack((labels_clear, labels_nan))

    pd.DataFrame(data=new_categ).to_csv(f'data/recovered/{filename}_naive/categ.csv', index=False)
    pd.DataFrame(data=new_cont).to_csv(f'data/recovered/{filename}_naive/cont.csv', index=False)
    pd.DataFrame(data=new_labels).to_csv(f'data/recovered/{filename}_naive/labels.csv', index=False)

if __name__ == '__main__':
    main(seed=42)
