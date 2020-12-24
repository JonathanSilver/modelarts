import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import json
import pickle


NUM_SAMPLES = 2001

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_feats(feats, num):
    idx = np.arange(len(feats))
    np.random.shuffle(idx)
    idx = idx[:num]
    assert len(idx) == num
    return np.sum(feats[idx], axis=0) / num


def preprocess_feats(feats):
    """
    feats (228, 4): nested lists of strings
    """
    feats = np.array(feats, np.float32)
    assert feats.shape[1:] == (228, 4)
    feats2 = []
    for feat in feats:
        feat[:, 0] /= 1800
        feat[:, 2] /= 500000
        feat = feat[:, 1:]
        feats2.append(feat)
    feats = np.array(feats2, np.float32)
    assert feats.shape[1:] == (228, 3)
    return feats


def get_pca(pca_path):
    with open(pca_path, 'rb') as file:
        return pickle.load(file)

class Preprocessor:
    
    def __init__(self, pca_path):
        self.pca = get_pca(pca_path)
    
    def __call__(self, data):
        """
        data = {
            key: feats (20, 228, 4)
        }
        """
        preprocessed_data = {}
        for name, feats in data.items():
            feats = preprocess_feats(feats)
            X = [sample_feats(feats, 15).reshape(-1) for _ in range(NUM_SAMPLES)]
            X = np.array(X, dtype=np.float32)
            assert X.shape == (NUM_SAMPLES, 228 * 3)
            X = self.pca.transform(X)
            assert X.shape == (NUM_SAMPLES, 228)
            preprocessed_data[name] = torch.tensor(X).to(device)
        return preprocessed_data


def make_predictions(pred):
    """
    pred (BATCH_SIZE, 15, 2): tensor
    """
    y_pred = []
    n = pred.shape[0]
    assert n % NUM_SAMPLES == 0 and pred.shape[1:] == (15, 2)
    pred = torch.argmax(pred, dim=-1)
    assert pred.shape[1:] == (15, )
    for k in range(0, n, NUM_SAMPLES):
        a = torch.sum(pred[k:k + NUM_SAMPLES] == 0)
        b = torch.sum(pred[k:k + NUM_SAMPLES] == 1)
        assert a != b
        y_pred.append(0 if a > b else 1)
    return y_pred


class Postprocessor:
    
    def __call__(self, data):
        """
        data = {
            key: pred (NUM_SAMPLES, 15, 2), tensor
        }
        """
        results = {}
        for name, pred in data.items():
            assert pred.shape == (NUM_SAMPLES, 15, 2)
            result = make_predictions(pred)
            assert len(result) == 1
            result = result[0]
            assert result in [0, 1]
            results[name] = 'cotton' if result == 0 else 'cotton_spandex'
        return results


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.models = nn.ModuleList([
                nn.Sequential(
                nn.Linear(228, 512),
                nn.Sigmoid(),
                nn.LayerNorm(512),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.Sigmoid(),
                nn.LayerNorm(512),
                nn.Dropout(),
                nn.Linear(512, 256),
                nn.Sigmoid(),
                nn.LayerNorm(256),
                nn.Dropout(),
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.LayerNorm(128),
                nn.Dropout(),
                nn.Linear(128, 2),
                nn.Softmax(dim=-1)
            ) for _ in range(15)
        ])

    def forward(self, pca_X):
        pred = [model(pca_X).unsqueeze(0) for model in self.models]
        pred = torch.cat(pred)  # (15, BATCH_SIZE, 2)
        pred = pred.transpose(0, 1)  # (BATCH_SIZE, 15, 2)
        assert pred.shape[1:] == (15, 2)
        return pred


def get_data(filename):
    with open(filename, encoding='utf-8') as file:
        return json.load(file)


def sample_from_record(record, num_per_sample, X, y):
    X.append(sample_feats(record['feats'], num_per_sample).reshape(-1))
    y.append(0 if record['name'] == 'cotton' else 1)


def repetitive_sample_from_record(record, num_samples, num_per_sample, X, y):
    [
        sample_from_record(record, num_per_sample, X, y)
        for _ in range(num_samples)
    ]


def get_id_dict(dataset):
    id_dict = {'cotton': [], 'cotton_spandex': []}
    for i, record in enumerate(dataset):
        id_dict[record['name']].append(i)
    return id_dict


def sample_dataset(dataset, num_samples, num_per_sample, balance):
    id_dict = get_id_dict(dataset)
    X = []
    y = []
    for record in dataset:
        repetitive_sample_from_record(record, num_samples, num_per_sample, X, y)
    a = len(id_dict['cotton'])
    b = len(id_dict['cotton_spandex']) * 1.0
    b = int(b)
    print(a, b)
    if balance and a != b:
        if a > b:
            diff = a - b
            idx = np.arange(b)
            name = 'cotton_spandex'
        else:  # b > a
            diff = b - a
            idx = np.arange(a)
            name = 'cotton'
        np.random.shuffle(idx)
        idx = idx[:diff]
        [
            repetitive_sample_from_record(
                dataset[id_dict[name][i]],
                num_samples, num_per_sample, X, y
            ) for i in idx
        ]
    return np.array(X), np.array(y, dtype=np.int64)


def process_dataset(dataset, num_samples=NUM_SAMPLES, num_per_sample=15, balance=False, dev=0):
    id_dict = get_id_dict(dataset)
    for record in dataset:
        record['feats'] = preprocess_feats(record['feat'])
        record.pop('feat')
    if dev == 0:  # no dev set
        return sample_dataset(dataset, num_samples, num_per_sample, balance)
    else:
        def sample_id(id_list):
            idx = np.arange(len(id_list))
            np.random.shuffle(idx)
            return [id_list[i] for i in idx[:dev]], [id_list[i] for i in idx[dev:]]
        c_dev, c_train = sample_id(id_dict['cotton'])
        cs_dev, cs_train = sample_id(id_dict['cotton_spandex'])
        dev_id = c_dev + cs_dev
        train_id = c_train + cs_train
        dev_ds = [dataset[i] for i in dev_id]
        train_ds = [dataset[i] for i in train_id]
        return [d for d in sample_dataset(train_ds, num_samples, num_per_sample, balance)] + \
               [d for d in sample_dataset(dev_ds, num_samples, num_per_sample, balance)]


def run_test(model, data_X, y_pred):
    data_loader = DataLoader(data_X, batch_size=512)
    model.eval()
    with torch.no_grad():
        for batch_X in data_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            y_pred = torch.cat([y_pred, output])
    return y_pred


def predict_on_single_model(model, data_X):
    y_pred = torch.zeros(0, 2).to(device)
    y_pred = run_test(model, data_X, y_pred)
    y_pred = torch.argmax(y_pred, dim=-1)
    return y_pred.cpu().numpy()


def evaluate(model, data_X, y):
    y_true = []
    assert len(y) % NUM_SAMPLES == 0
    for k in range(0, len(y), NUM_SAMPLES):
        assert np.sum(y[k:k + NUM_SAMPLES] == y[k]) == NUM_SAMPLES
        y_true.append(y[k])
    y_pred = torch.zeros(0, 15, 2).to(device)
    y_pred = run_test(model, data_X, y_pred)
    y_pred = make_predictions(y_pred)
    assert len(y_true) == len(y_pred)
    print(classification_report(y_true, y_pred, digits=4))


def train_single_model(model, pca_train_X, train_y, pca_dev_X, dev_y, pca_test_X, test_y):
    n = len(pca_train_X)
    idx = np.arange(n)
    idx = np.random.choice(idx, size=int(n * 0.9))
    training_set = DataLoader(list(zip(pca_train_X[idx], train_y[idx])),
                              batch_size=512,
                              shuffle=True)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()
    epoch = 0
    best_acc, best_epoch = 0, 0
    while epoch < 100:
        epoch += 1
        model.train()
        with torch.enable_grad():
            for batch_X, batch_y in training_set:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_X)
                loss = loss_fn(output, torch.eye(2).to(device)[batch_y])
                loss.backward()
                optimizer.step()
        acc = np.sum(predict_on_single_model(model, pca_dev_X) == dev_y)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        elif epoch - best_epoch >= 10:
            print('early stopping on epoch {}, dev acc {:.4f}'.format(epoch, acc / dev_y.shape[0]))
            break
    y_pred = predict_on_single_model(model, pca_test_X)
    return np.sum(y_pred == test_y) / y_pred.shape[0]


def train(train_json_file, test_json_file, pca_path, model_path):
    models = Classifier()
    models.to(device)
    
    dataset_train = get_data(train_json_file)
    dataset_test = get_data(test_json_file)

    train_X, train_y, dev_X, dev_y = process_dataset(dataset_train, balance=True, dev=10)
    test_X, test_y = process_dataset(dataset_test)

    del dataset_train, dataset_test

    pca = PCA(n_components=228)
    pca_train_X = pca.fit_transform(train_X)
    pca_dev_X = pca.transform(dev_X)
    pca_test_X = pca.transform(test_X)
    
    del train_X, dev_X, test_X
    
    print('\nsaving PCA to: ' + pca_path)
    with open(pca_path, 'wb') as file:
        pickle.dump(pca, file)
    
    print('\ndata preprocessed\n\ntraining models\n')

    for i, model in enumerate(models.models, start=1):
        acc = train_single_model(model,
                                 pca_train_X, train_y,
                                 pca_dev_X, dev_y,
                                 pca_test_X, test_y)
        print(f'{i}/{len(models.models)} - test acc: {acc:.4f}')
    
    print('\nmodels trained\n\nsaving models to ' + model_path)
    torch.save(models.state_dict(), model_path)

    print('\nevaluating on the training set:')
    evaluate(models, pca_train_X, train_y)
    print('\nevaluating on the dev set:')
    evaluate(models, pca_dev_X, dev_y)
    print('\nevaluating on the test set:')
    evaluate(models, pca_test_X, test_y)


def test(test_json_file, pca_path, model_path):
    dataset_test = get_data(test_json_file)
    test_X, test_y = process_dataset(dataset_test)

    del dataset_test

    print('loading PCA from: ' + pca_path)
    pca = get_pca(pca_path)

    pca_test_X = pca.transform(test_X)

    del pca, test_X

    models = Classifier()
    models.to(device)
    print('loading models from: ' + model_path)
    models.load_state_dict(torch.load(model_path))

    print('evaluating on the test set:')
    evaluate(models, pca_test_X, test_y)

    del models, pca_test_X, test_y


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser

    np.random.seed(2020)
    torch.manual_seed(2020)

    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, help="data directory containing 'train.json' and 'test.json'.", default='.')
    parser.add_argument('--model-dir', type=str, help="model directory to store 'classifier-pca.pkl' and 'classifier.pth'.", default='.')
    args, _ = parser.parse_known_args()

    DATA_DIR = args.data_dir
    PCA_PATH = os.path.join(args.model_dir, 'classifier-pca.pkl')
    MODEL_PATH = os.path.join(args.model_dir, 'classifier.pth')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    train(os.path.join(DATA_DIR, 'train.json'),
          os.path.join(DATA_DIR, 'test.json'),
          PCA_PATH, MODEL_PATH)
    test(os.path.join(DATA_DIR, 'test.json'),
         PCA_PATH, MODEL_PATH)
