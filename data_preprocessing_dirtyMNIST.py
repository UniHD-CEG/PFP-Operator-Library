import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
from ddu_dirty_mnist import FastMNIST, DirtyMNIST, AmbiguousMNIST


def prepare_data(train_dataset_name, test_dataset_names, make_dataloaders: bool = True, batch_size: int = 100,
                 filter_cls: int = None, device="cpu"):
    """
    Prepares MNIST datasets as specified
    :param train_dataset_name: name of train_dataset
    :param test_dataset_names: name of test_dataset(s)
    :param make_dataloaders: whether to provide data in dataloaders or tensors
    :param batch_size: batch size for dataloaders
    :param filter_cls: for pure MNIST only: class to filter out during training
    :param device: device for the dataset
    :return: train_dataset, test_dataset(s) in a dict
    """

    train_dataset = load_data(train_dataset_name, train=True, filter_cls=filter_cls, make_dataloaders=make_dataloaders,
                              batch_size=batch_size, device=device)
    test_datasets = {}
    for name in test_dataset_names:
        dataset = load_data(name, train=False, filter_cls=filter_cls, make_dataloaders=make_dataloaders,
                            batch_size=batch_size, device=device)
        if filter_cls and name == "MNIST":
            test_datasets[name + "_filtered"] = dataset[0]
            test_datasets[name + "_filtered_cls"] = dataset[1]
        else:
            test_datasets[name] = dataset

    return train_dataset, test_datasets


def load_data(dataset_name: str = "MNIST", train: bool = True, filter_cls: int = None, make_dataloaders: bool = True,
              batch_size: int = 100, device="cpu"):
    """
    Loads dataset as specified
    :param dataset_name: name of the dataset to load
    :param train: whether to load train/test data
    :param filter_cls: for pure MNIST only: class to filter out during training
    :param make_dataloaders: whether to provide data in dataloaders or tensors
    :param batch_size: batch size for dataloaders
    :param device: device for the dataset
    :return: dataset: dataloader or tuple (x,y)
    """

    if dataset_name == "MNIST":
        # FastMNIST for perfomance
        dataset = FastMNIST("./data/", train=train, normalize=True, download=True, noise_stddev=0.0, device=device)

        # only for pure MNIST
        if filter_cls is not None:
            return filter_class(filter_cls, dataset, train, make_dataloaders, batch_size)

    elif dataset_name == "AmbiguousMNIST":
        dataset = AmbiguousMNIST("./data/", train=train, normalize=True, download=True, noise_stddev=0.05,
                                 device=device)
        if not train:
            dataset = remove_duplicates_ambiguous_mnist(dataset)

    elif dataset_name == "FashionMNIST":
        dataset = FastFashionMNIST("./data/", train=train, normalize=True, download=True, device=device)

    else:
        dataset = DirtyMNIST("./data/", train=train, normalize=True, download=True, noise_stddev=0.05,
                             device=device)
        if not make_dataloaders:
            x = torch.concat([dataset.datasets[0].data, dataset.datasets[1].data])
            y = torch.concat([dataset.datasets[0].targets, dataset.datasets[1].targets])
            return x, y

    if make_dataloaders:
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True)
    else:
        return dataset.data, dataset.targets


def filter_class(filter_cls: int, dataset: Dataset, train: bool = True, make_dataloaders: bool = True,
                 batch_size: int = 100):
    """

    :param filter_cls: for pure MNIST only: class to filter out during training
    :param dataset: dataset to filter
    :param train: whether train or test data is processed
    :param make_dataloaders: whether to provide data in dataloaders or tensors
    :param batch_size: batch size for dataloaders
    :return: dataset: dataloader(s) or tuple(s) (x,y)
    """
    # prepare indices
    idx_filter = torch.nonzero(dataset.targets != filter_cls).squeeze()  # squeeze to remove [:, 1] dim
    idx_filter_cls_only = torch.nonzero(dataset.targets == filter_cls).squeeze()

    if make_dataloaders:
        dataset_filtered = Subset(dataset, idx_filter)
        loader_filtered = DataLoader(dataset_filtered, batch_size=batch_size, shuffle=train)

        if not train:  # only need filtered cls during evaluation
            dataset_filter_cls_only = Subset(dataset, idx_filter_cls_only)
            loader_filtered_cls = DataLoader(dataset_filter_cls_only, batch_size=batch_size, shuffle=train)
            return loader_filtered, loader_filtered_cls

        return loader_filtered

    else:
        x_filtered = dataset.data[idx_filter]
        y_filtered = dataset.targets[idx_filter]

        if not train:  # only need filtered cls during evaluation
            x_filtered_cls_only = dataset.data[idx_filter_cls_only]
            y_filtered_cls_only = dataset.targets[idx_filter_cls_only]
            return (x_filtered, y_filtered), (x_filtered_cls_only, y_filtered_cls_only)

        return x_filtered, y_filtered


class FastFashionMNIST(torchvision.datasets.FashionMNIST):
    """
    Like FastMNIST
    """

    def __init__(self, *args, normalize, device, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = self.data.unsqueeze(1).float().div(255)

        self.data, self.targets = self.data.to(device), self.targets.to(device)

        if normalize:
            self.data = self.data.sub_(0.2860).div_(0.3530)

    def __getitem__(self, index: int):

        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def remove_duplicates_ambiguous_mnist(dataset):
    """
    Remove duplicate test picture-label pairs from ambiguous mnist: the authors sampled 10 labels for each of the 6k
    pictures, where the 10 labels are not necessarily unique -
    while it makes sense during training to keep the oversampled pictures, this will unnecessarily inflate the
    test dataset statistics since the same picture-label combination will appear multiple times
    :param dataset: ambiguous mnist dataset
    :return: cleaned_dataset
    """
    data = dataset.data.reshape(dataset.targets.size(0) // 10, 10, 28, 28)
    targets = dataset.targets.reshape(dataset.targets.size(0) // 10, 10)

    clean_data = []
    clean_targets = []
    for sample_idx in range(data.size(0)):
        unique_targets = torch.unique(targets[sample_idx, :])
        clean_targets.append(unique_targets)
        clean_data.append(data[sample_idx, :unique_targets.size(0), :, :])

    clean_data = torch.cat(clean_data, dim=0).unsqueeze(1)
    clean_targets = torch.cat(clean_targets, dim=0)
    dataset.data = clean_data
    dataset.targets = clean_targets

    return dataset


if __name__ == "__main__":
    prepare_data(train_dataset_name="MNIST", test_dataset_names=["MNIST", "AmbiguousMNIST"], filter_cls=4)
