import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import glob


def get_data_loaders(batch_size_train, batch_size_test, directory_path, directory_path_test=None, num_of_subjects=1):
    print(directory_path)
    np_dataset = []
    for idx, np_name in enumerate(glob.glob(directory_path + '/*.np[yz]')):
        if idx >= num_of_subjects:
            print(f"Loaded {num_of_subjects} subjects")
            break
        print(f"loading {np_name}")
        np_dataset.append(np_name)

    np_dataset_test = None
    if directory_path_test is not None:
        np_dataset_test = []
        for idx, np_name in enumerate(glob.glob(directory_path_test + '/*.np[yz]')):
            if idx >= num_of_subjects:
                print(f"Loaded {num_of_subjects} subjects for test")
                break
            print(f"loading {np_name}")
            np_dataset_test.append(np_name)

    train_loader, test_loader = data_generator_np(subject_files=np_dataset,
                                                  subject_files_test=np_dataset_test,
                                                  batch_size_train=batch_size_train,
                                                  batch_size_test=batch_size_test)

    names = np_dataset + np_dataset_test if np_dataset_test is not None else np_dataset
    
    return train_loader, test_loader, names


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


def data_generator_np(subject_files, batch_size_train, batch_size_test, subject_files_test=None, train_test_ratio=0.8):
    shuffle_dataset = True

    # Loading numpy dataset
    dataset = LoadDataset_from_numpy(subject_files)
    if subject_files_test is not None:
        # Split is between files of test and train
        dataset_test = LoadDataset_from_numpy(subject_files_test)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=batch_size_train,
                                                   drop_last=False,
                                                   num_workers=0)

        test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=batch_size_test,
                                                  drop_last=False,
                                                  num_workers=0)
    else:
        # Creating data indices for training and test splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor((1 - train_test_ratio) * dataset_size))
        if shuffle_dataset:
            np.random.shuffle(indices)

        train_indices, test_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=batch_size_train,
                                                   sampler=train_sampler,
                                                   drop_last=False,
                                                   num_workers=0)

        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size_test,
                                                  sampler=test_sampler,
                                                  drop_last=False,
                                                  num_workers=0)

    print(f"train_loader length in batches: {len(train_loader)}")
    print(f"test_loader length in batches: {len(test_loader)}")

    return train_loader, test_loader
