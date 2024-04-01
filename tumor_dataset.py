# TODO: Brats Training 355 seg file odd?

import glob

import nibabel as nib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class TumorDataset(Dataset):
    def __init__(self, img_dir, transform=False):
        self.img_dir = img_dir
        # TODO: Transformations
        # self.transform = v2.Compose([
        #     v2.Normalize(mean=(100.9358), std=(torch.sqrt(233677.6719)))
        # ])

    def __len__(self):
        return 100 # TODO: Remove
        return len(glob.glob(f'{self.img_dir}/*/'))

    def __getitem__(self, idx):
        """
        Returns [image_native, image_t1weighted, image_t2weighted, image_flair, image_seg]
        """
        images = []
        for img_type in ['t1', 't1ce', 't2', 'flair', 'seg']:
            img = nib.load(f'{self.img_dir}/BraTS20_Training_{idx + 1:0{3}}/BraTS20_Training_{idx + 1:0{3}}_{img_type}.nii').get_fdata()
            img = torch.Tensor(img)

            if img_type == 'seg':
                img = img.long()
                img[img == 4] = 3 # TODO: Pre-process this

            # TODO: transformations
            # img = self.transform(img)
            images.append(img)

        X, y = torch.stack(images[:-1], dim=0), images[-1]

        return X, y