import torch
from tqdm import tqdm
import xgboost as xgb

from evaluation.evaluate import diceForEach


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_dataset, name, load=False):
    if load:
        print('Loading Booster')
        model = xgb.Booster()
        model.load_model(f'xg/models/{name}.json')
        return model

    print('Training Booster')
    model = None
    param = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 4,
        'device': device,
        'verbosity': 2,
        'eta': 0.3,
        'gamma': 0,
        'max_depth': 6,
    }

    for d in tqdm(range(0, len(train_dataset))):
        if model:
            param.update({
                'process_type': 'update',
                'updater': 'refresh',
            })

        x, y = train_dataset[d]

        x = x.reshape(-1, 4)
        y = y.reshape(-1)

        data = xgb.DMatrix(x, label=y)

        model = xgb.train(param, dtrain=data, xgb_model=model)

    print(f'Saving booster to xg/models/{name}.json')
    model.save_model(f'xg/models/{name}.json')

    return model


def evaluate(model, val_dataset):
    print('Evaluating Booster')
    results = {}

    count = 0
    for X, y in tqdm(val_dataset):
        h, w, d, _ = X.shape
        h, w, d = y.shape

        X = X.reshape(-1, 4)
        y = y.reshape(-1)

        y_hat = torch.Tensor(model.predict(xgb.DMatrix(X)))

        sub_results = diceForEach(y_hat, y, classes=[0, 1, 2, 3])

        for key in sub_results:
            if key not in results:
                results[key] = sub_results[key]
            else:
                results[key] = (results[key] * count + sub_results[key]) / (count + 1)
        count += 1

    return results

