import torch
import numpy as np
from PIL import Image
import h5py, glob, random
from torch.utils.data import Dataset

class Dataset_from_syn(Dataset):
    def __init__(self, src_data, sigma, gray=False, transform=None):
        self.data = glob.glob(src_data)
        self.gray = gray
        self.sigma = sigma/255.
        self.transform = transform

    def __getitem__(self, index):
        clean = Image.open(self.data[index])
        if self.gray:
            clean = clean.convert('L')
        if self.transform:
            clean = self.transform(clean)
        noise = torch.normal(torch.zeros(clean.size()), self.sigma)
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return noisy, clean

    def __len__(self):
        return len(self.data)


class Dataset_for_eval(Dataset):
    def __init__(self, src_path, gray=False):
        self.gt_data = glob.glob(src_path[0])
        self.noisy_data = glob.glob(src_path[1])
        self.gt_data.sort()
        self.noisy_data.sort()
        self.gray = gray
        assert len(self.gt_data)==len(self.noisy_data)

    def __getitem__(self, index):
        noisy = np.array(Image.open(self.noisy_data[index]))/255.0
        clean = np.array(Image.open(self.gt_data[index]))/255.0
        noisy = noisy[0:(noisy.shape[0]//8)*8, 0:(noisy.shape[1]//8)*8]
        clean = clean[0:(clean.shape[0]//8)*8, 0:(clean.shape[1]//8)*8]
        if self.gray:
            noisy = np.expand_dims(noisy, -1)
            clean = np.expand_dims(clean, -1)
        noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(noisy, (2, 0, 1)))).float()
        clean = torch.from_numpy(np.ascontiguousarray(np.transpose(clean, (2, 0, 1)))).float()
        return noisy, clean

    def __len__(self):
        return len(self.noisy_data)


class Dataset_h5_real(Dataset):
    def __init__(self, src_path, patch_size):
        self.path = src_path
        self.patch_size = patch_size
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        data = np.array(h5f[self.keys[index]])
        h5f.close()

        rnd_h = random.randint(0, data.shape[0]-self.patch_size)
        rnd_w = random.randint(0, data.shape[1]-self.patch_size)
        patch = data[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size]

        if random.random() > 0.5: #RandomRot90
            patch = patch.transpose(1, 0, 2)
        if random.random() > 0.5: #RandomHorizontalFlip
            patch = patch[:, ::-1, :]
        if random.random() > 0.5: #RandomVerticalFlip
            patch = patch[::-1, :, :]
        
        patch = np.clip(patch.astype(np.float32)/255., 0.0, 1.0)
        noisy = torch.from_numpy(np.ascontiguousarray(
            np.transpose(patch[:, :, 0:3], (2, 0, 1))
        )).float()
        clean = torch.from_numpy(np.ascontiguousarray(
            np.transpose(patch[:, :, 3:6], (2, 0, 1))
        )).float()

        return noisy, clean

    def __len__(self):
        return len(self.keys)

if __name__=='__main__':
    dataset = Dataset_h5_real('dataset/train_real/train.h5', 128)
    dataset.__getitem__(0)