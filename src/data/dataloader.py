import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X = np.load(np_dataset[0])["x"]
        y = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X = np.vstack((X, np.load(np_file)["x"]))
            y = np.append(y, np.load(np_file)["y"])

        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(subject_files, batch_size, train_test_ratio=0.8):

    shuffle_dataset = True
    random_seed = 42

    # Loading numpy dataset
    dataset = LoadDataset_from_numpy(subject_files)

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor((1 - train_test_ratio) * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              sampler=test_sampler,
                                              drop_last=False,
                                              num_workers=0)

    print(f"train_loader length in batches: {len(train_loader)}")
    print(f"test_loader length in batches: {len(test_loader)}")

    return train_loader, test_loader






