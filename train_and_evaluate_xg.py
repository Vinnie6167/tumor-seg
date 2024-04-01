from torch.utils.data import random_split
from tumor_dataset import TumorDataset
from booster.model import evaluate, train


def main():
    img_dir = 'data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
    train_ds, val_ds = random_split(TumorDataset(img_dir=img_dir, transform=True), (0.8, 0.2))
    print(f'Dataset loaded: {img_dir}, train length: {len(train_ds)}, validation length: {len(val_ds)}')

    model = train(train_ds, name='test', load=False)

    eval = evaluate(model, val_ds)

    print(eval)


if __name__ == '__main__':
    main()