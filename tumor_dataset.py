import glob

import nibabel as nib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class TumorDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # self.transform = v2.Compose([
        #     v2.Normalize(mean=(,), std=(,))
        # ])

    def __len__(self):
        return len(glob.glob(f'{self.img_dir}/*/'))

    def __getitem__(self, idx):
        """
        Returns [image_native, image_t1weighted, image_t2weighted, image_flair, image_seg]
        """
        res = []
        for img_type in ['t1', 't1ce', 't2', 'flair', 'seg']:
            img = nib.load(f'{self.img_dir}/BraTS20_Training_{idx + 1:0{3}}/BraTS20_Training_{idx + 1:0{3}}_{img_type}.nii').get_fdata()
            img = torch.Tensor(img)
            # img = self.transform(img)
            res.append(img)

        return res