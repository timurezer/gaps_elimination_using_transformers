import numpy as np

from mlm import *

def count_acc(y_true, y_pred):
    x = (torch.round(nn.Sigmoid()(y_pred)) == y_true).sum().float().item() / y_true.shape[0] / y_true.shape[1]
    return x

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

def my_subplots(train_loss, val_loss, train_acc, val_acc, test_loss, test_acc):
    _, ax1 = plt.subplots()
    plot_results(ax1, train_loss, val_loss, test_loss, 'Loss')
    _, ax2 = plt.subplots()
    plot_results(ax2, train_acc, val_acc, test_acc, 'Accuracy')
    plt.show()

class SingleDataset(Dataset):
    """
    generate nans (masks), labels - masked values
    Генерация в датасете - более естественно, т.к. forward в модели
    """

    def __init__(self, X: np.array):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]


def simple_train_test_split(X, test_size=0.2, val_size=0.05, random_state=42):
    n1 = int(np.around(test_size * X.shape[0]))
    n2 = int(np.around((val_size + test_size) * X.shape[0]))
    if random_state is not None:
        np.random.seed(random_state)
    idx = np.random.permutation(X.shape[0])
    X_test = X[idx][:n1]
    X_val = X[idx][n1:n2]
    X_train = X[idx][n2:]
    return X_train, X_val, X_test


def mlm_single_pass(model, dataloader, loss_func, device, optim=None):
    loss_count, acc_count = 0, 0
    for i, x_categ in enumerate(dataloader):
        x_categ = x_categ.long().to(device)
        pred, labels = model.forward(x_categ)
        labels = labels.long().to(device)
        if loss_func is not None:
            loss = loss_func(pred, labels)
            loss_count += loss.item()
            acc_count += count_acc(labels, pred)
        if optim is not None:
            loss.backward()
            optim.step()
    return loss_count / len(dataloader), acc_count / len(dataloader)

def train_model(model, loss, optim, epochs, device, dataloaders, single_pass):
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    train_loss_all, val_loss_all, train_acc_all, val_acc_all = [], [], [], []
    # training loop
    for epoch in range(epochs):
        # train
        train_loss, train_acc = single_pass(model, dataloader_train, loss, device, optim)
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
    model.train()
    return train_loss_all, val_loss_all, train_acc_all, val_acc_all, test_loss, test_acc

def train_mlm(model, x_categ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch_size = 1900
    epochs = 20  # 10
    lr = 1e-4
    optim = Adam(model.parameters(), lr=lr)
    loss = F.binary_cross_entropy_with_logits
    X_train, X_val, X_test = simple_train_test_split(x_categ)

    dataloader_train = DataLoader(SingleDataset(X_train), batch_size=batch_size)
    dataloader_val = DataLoader(SingleDataset(X_val), batch_size=64)
    dataloader_test = DataLoader(SingleDataset(X_test), batch_size=64)

    dataloaders = [dataloader_train, dataloader_val, dataloader_test]
    metrics = train_model(model, loss, optim, epochs, device, dataloaders, single_pass=mlm_single_pass)
    return metrics

def predict(model, x_categ):
    model.inference = True
    model.eval()
    dataloader = DataLoader(SingleDataset(x_categ), batch_size=64)
    res = []
    for X in dataloader:
        pred, _ = model.forward(X)
        recovered_labels = torch.argmax(pred, dim=1).detach().cpu().numpy().reshape(-1,1)
        categs = model.categories_offset.detach().cpu().numpy() - model.num_special_tokens
        diff = (recovered_labels.reshape(-1, 1) - categs.reshape(1, -1)).astype(float)
        diff[diff < 0] = float('inf')
        recovered_labels = np.min(diff, axis=1).astype(int)
        res.extend(recovered_labels)
    return np.array(res)

def main(mask_mode='different', seed=42):
    filename = sys.argv[1]
    path = os.path.join('data/preprocessed', filename)
    path_nans = os.path.join('data/with_nans', filename)
    data_categ_all = pd.read_csv(os.path.join(path_nans, 'categ.csv')).to_numpy()  # read with nans file
    true_labels = pd.read_csv(os.path.join(path_nans, 'true_labels.csv')).to_numpy().reshape(-1)
    data_cont = pd.read_csv(os.path.join(path, 'cont.csv')).to_numpy()
    data_labels = pd.read_csv(os.path.join(path, 'labels.csv')).to_numpy()

    categ_clear, categ_nan, cont_clear, cont_nan, labels_clear, labels_nan, nans_pos = \
        separate_nans(data_categ_all, data_cont, data_labels)  # data_categ is pure of NaNs

    categories = tuple(len(np.unique(categ_clear[:, i])) for i in range(categ_clear.shape[1]))
    model = TabTransformerMLM(
        categories=categories,  # tuple containing the number of unique values within each category
        dim=32,  # dimension, paper set at 32
        depth=6,  # depth, paper recommended 6
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        seed=seed,
        mask_mode=mask_mode,    # single/different
        inference=False

    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logs = train_mlm(model, categ_clear)
    my_subplots(*logs)

    recovered_labels = predict(model, categ_nan)
    print(recovered_labels[:10])
    print('acc:', accuracy_score(true_labels, recovered_labels))
    categ_nan[nans_pos] = recovered_labels
    new_categ = np.vstack((categ_clear, categ_nan))
    new_cont = np.vstack((cont_clear, cont_nan))
    new_labels = np.vstack((labels_clear, labels_nan))


    pd.DataFrame(data=new_categ).to_csv(f'data/recovered/{filename}_mlm_{mask_mode}/categ.csv', index=False)
    pd.DataFrame(data=new_cont).to_csv(f'data/recovered/{filename}_mlm_{mask_mode}/cont.csv', index=False)
    pd.DataFrame(data=new_labels).to_csv(f'data/recovered/{filename}_mlm_{mask_mode}/labels.csv', index=False)


if __name__ == '__main__':
    main(mask_mode='different', seed=42)

