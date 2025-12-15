# -*- coding: utf-8 -*-
import copy
import logging
import random

import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy


from datasets.common import train_label_split, get_random_indices, \
    get_labeled_loader, get_target_indices, image_dataset_with_indices, poison_image_dataset
from datasets.image_dataset import ImageDataset
from datasets.multi_image_dataset import MultiImageDataset


# transform for CIFAR train dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

# transform for CIFAR test dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_labeled_data_with_2_party(data_dir, dataset, dtype="Train"):
    """
    read data from local file

    :param data_dir: dir path of local file
    :param str dataset: dataset name, support cifar10 and cifar100
    :param str dtype: read "Train" or "Test" data
    :return: tuple containing X and Y
    """
    train = True if dtype == 'Train' else False
    transform = train_transform if dtype == 'Train' else test_transform
    if dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train,
                                               download=True, transform=transform)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=train,
                                                download=True, transform=transform)
    all_data = dataset.data
    return all_data, np.array(dataset.targets)


def load_two_party_data(data_dir, args):
    """
    get data from local dataset, only support two parties

    :param data_dir: path of local dataset
    :param args: configuration
    :return: tuple contains:
        (1) X_train: normal train features;
        (2) y_train: normal train labels;
        (3) X_test: normal test features;
        (4) y_test: normal test labels;
        (5) backdoor_y_train: backdoor train labels;
        (6) backdoor_X_test: backdoor test features;
        (7) backdoor_y_test: backdoor test labels;
        (8) backdoor_indices_train: indices of backdoor samples in normal train dataset;
        (9) backdoor_target_indices: indices of backdoor label in normal train dataset;
        (10) train_labeled_indices: indices of labeled samples in normal train dataset;
        (11) train_unlabeled_indices: indices of unlabeled samples in normal train dataset
        (12) backdoor_X_train: train dataset that backdoor_target_indices sample had been replaced with backdoor_indices sample in adv
    """
    logging.info("# load_two_party_data")
    # read train data from local file
    X, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                         dataset=args['dataset'],
                                         dtype='Train')
    n_train = args['target_train_size']
    n_test = args['target_test_size']
    if n_train != -1:
        indices = get_random_indices(n_train, len(X))
        X_train, y_train = X[indices], y[indices]
    else:
        X_train, y_train = X, y

    # read test data from local file
    X_test, y_test = get_labeled_data_with_2_party(data_dir=data_dir,
                                                   dataset=args['dataset'],
                                                   dtype='Test')

    if n_test != -1:
        indices = get_random_indices(n_test, len(X_test))
        X_test, y_test = X_test[indices], y_test[indices]

    # randomly select samples of other classes from normal train dataset as backdoor samples to generate backdoor train dataset
    train_indices = np.where(y_train != args['backdoor_label'])[0]
    backdoor_indices_train = np.random.choice(train_indices, args['backdoor_train_size'], replace=False)
    backdoor_y_train = copy.deepcopy(y_train)
    backdoor_y_train[backdoor_indices_train] = args['backdoor_label']

    # randomly select samples of other classes from normal test dataset to generate backdoor test dataset
    test_indices = np.where(y_test != args['backdoor_label'])[0]
    backdoor_indices_test = np.random.choice(test_indices, args['backdoor_test_size'], replace=False)
    backdoor_X_test, backdoor_y_test = X_test[backdoor_indices_test], \
                                       y_test[backdoor_indices_test]
    backdoor_y_test_true = backdoor_y_test
    backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])

    # split labeled and unlabeled samples in normal train dataset, for LR-BA
    train_labeled_indices, train_unlabeled_indices = \
        train_label_split(y_train, args['train_label_size'], args['num_classes'],
                          args['train_label_non_iid'], args['backdoor_label'], args['train_label_fix_backdoor'])

    # randomly select samples of backdoor label in normal train dataset, for gradient-replacement
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['train_label_size'])

    index = [i for i in range(len(X_train))]
    sr_X_train = []
    sr_y_train = []
    np.random.shuffle(index)
    for i in range(0, len(X_train)):
        sr_X_train.append(X_train[index[i]])
        sr_y_train.append(y_train[index[i]])
    sr_X_train = np.array(sr_X_train)
    sr_y_train = np.array(sr_y_train)
    temp = [i for i in range(len(sr_X_train) - 500, len(sr_X_train))]
    sr_ba_backdoor_target_indices = get_target_indices(sr_y_train, args['backdoor_label'], args['backdoor_train_size'], backdoor_indices=temp)

    sr_ba_backdoor_target_indices.sort()
    sr_X_train, sr_ba_backdoor_indices_train = poison_image_dataset(sr_X_train, sr_y_train, sr_ba_backdoor_target_indices, args)

    indices_dict = None
    if args['trigger'] == 'feature' and not args['trigger_add'] and args['idea'] == 2:
        n = int(args['backdoor_train_size'] / args['num_classes'])
        indices_dict = {}
        for i in range(args['num_classes']):
            indices = np.where(sr_y_train == i)[0]
            np.random.shuffle(indices)
            indices_dict[i] = indices[:n]


    logging.info("y_train.shape: {}".format(y_train.shape))
    logging.info("y_test.shape: {}".format(y_test.shape))
    logging.info("backdoor_y_test.shape: {}".format(backdoor_y_test.shape))

    labeled_y_train = y_train[train_labeled_indices]
    temp = []
    for i in range(args['num_classes']):
        indices = np.where(labeled_y_train == i)[0]
        temp.append(len(indices))
    # logging.info('labeled labels sum: {}, all: {}'.format(np.sum(temp), temp))
    logging.info('labeled labels sum: {}'.format(np.sum(temp)))

    return X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
           backdoor_indices_train, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices, sr_X_train, sr_y_train, sr_ba_backdoor_target_indices,sr_ba_backdoor_indices_train,\
           indices_dict, backdoor_y_test_true


def generate_dataloader(args, data_list, batch_size, transform, shuffle=True, backdoor_indices=None, trigger=None, trigger_add=None, source_indices=None):
    """
    generate loader from dataset

    :param tuple data_list: contains X and Y
    :param int batch_size: batch of loader
    :param transform: transform of loader
    :param bool shuffle: whether to shuffle loader
    :param backdoor_indices: indices of backdoor samples in normal dataset, add trigger when loading data if index is in backdoor_indices
    :param trigger: control whether add pixel trigger when load dataloader
    :return: loader
    """
    X, y = data_list

    if args['n_passive_party'] > 1:
        MultiImageDatasetWithIndices = image_dataset_with_indices(MultiImageDataset)
        party_num = args['n_passive_party'] + 1
        ds = MultiImageDatasetWithIndices(X, torch.tensor(y),
                                          transform=transform,
                                          backdoor_indices=backdoor_indices,
                                          party_num=party_num,
                                          trigger=trigger, trigger_add=trigger_add, source_indices=source_indices, adversary=args['adversary'])
        if args['backdoor'] == 'TECB':
            if party_num == 4:
                ds.pixel_pattern = torch.zeros(3, 16, 16)
            if party_num == 8:
                ds.pixel_pattern = torch.zeros(3, 16, 8)
    else:
        ImageDatasetWithIndices = image_dataset_with_indices(ImageDataset)
        # split x into halves for parties when loading data, only support two parties
        ds = ImageDatasetWithIndices(X, torch.tensor(y),
                                     transform=transform,
                                     backdoor_indices=backdoor_indices,
                                     half=16, trigger=trigger, trigger_add=trigger_add, source_indices=source_indices)
        if args['backdoor'] == 'TECB':
            ds.pixel_pattern = torch.zeros(3,32,16)

    dl = data.DataLoader(dataset=ds,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=False)
    return dl


def get_cifar_dataloader(args):
    """
    generate loader of CIFAR dataset, support cifar10 and cifar100

    :param args: configuration
    :return: tuple contains:
        (1) train_dl: loader of normal train dataset;
        (2) test_dl: loader of normal test dataset;
        (3) backdoor_train_dl: loader of backdoor train dataset, including normal and backdoor samples, used by data poisoning
        (4) backdoor_test_dl: loader of backdoor test dataset, only including backdoor samples, used to evaluate ASR
        (5) g_r_train_dl: loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
        (6) backdoor_indices: indices of backdoor samples in normal train dataset;
        (7) backdoor_target_indices: indices of backdoor label in normal train dataset, used by Gradient-Replacement
        (8) labeled_dl: loader of labeled samples in normal train dataset, used by LR-BA;
        (9) unlabeled_dl: loader of unlabeled samples in normal train dataset, used by LR-BA
    """
    # get dataset
    result = load_two_party_data("../../data/", args)
    X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices, sr_X_train, sr_y_train, sr_ba_backdoor_target_indices,\
    sr_ba_backdoor_indices_train, labeled_indices_dict, backdoor_y_test_true = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader(args, (X_train, y_train), batch_size, train_transform, shuffle=True)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader(args, (X_test, y_test), batch_size, test_transform, shuffle=False)

    backdoor_train_dl = generate_dataloader(args, (X_train, backdoor_y_train), batch_size, train_transform,
                                            shuffle=True,
                                            backdoor_indices=backdoor_indices)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader(args, (backdoor_X_test, backdoor_y_test), batch_size, test_transform,
                                           shuffle=False,
                                           backdoor_indices=np.arange(args['backdoor_test_size']), trigger=args['trigger'], trigger_add=args['trigger_add'])

    # get loader of labeled and unlabeled normal train dataset, used by LR-BA
    labeled_dl, unlabeled_dl = get_labeled_loader(train_dataset=train_dl.dataset,
                                                  labeled_indices=train_labeled_indices,
                                                  unlabeled_indices=train_unlabeled_indices,
                                                  args=args)

    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    # backdoor_indices: indices of none-target samples
    g_r_train_dl = generate_dataloader(args, (X_train, y_train), batch_size, train_transform,
                                       shuffle=True,
                                       backdoor_indices=backdoor_indices)


    if args['backdoor'] == 'sr_ba':
        s_r_train_dl = generate_dataloader(args, (sr_X_train, sr_y_train), batch_size, train_transform,
                                       shuffle=False,
                                       backdoor_indices=sr_ba_backdoor_target_indices, trigger=args['trigger'],
                                        trigger_add=args['trigger_add'], source_indices=sr_ba_backdoor_indices_train)
    elif args['backdoor'] in ['villain', 'splitNN']:
        s_r_train_dl = generate_dataloader(args, (sr_X_train, sr_y_train), batch_size, train_transform,
                                           shuffle=False,
                                           backdoor_indices=sr_ba_backdoor_target_indices, trigger=args['trigger'],
                                           trigger_add=args['trigger_add'], source_indices=None)
    elif args['backdoor'] in ['TECB']:
        s_r_train_dl = generate_dataloader(args, (sr_X_train, sr_y_train), batch_size, train_transform,
                                           shuffle=False,
                                           backdoor_indices=sr_ba_backdoor_target_indices, trigger=args['trigger'],
                                           trigger_add=args['trigger_add'], source_indices=None)
        s_r_train_dl.dataset.CBP_backdoor_indices = sr_ba_backdoor_target_indices

        list_c = []
        length = len(sr_X_train)
        total_batches = (length + batch_size - 1) // batch_size
        for batch in range(total_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, length)
            current_b_indices = [idx for idx in sr_ba_backdoor_target_indices if start_idx <= idx < end_idx]
            num_b_in_batch = len(current_b_indices)

            non_b_indices = [idx for idx in range(start_idx, end_idx) if idx not in sr_ba_backdoor_target_indices]

            if num_b_in_batch > 0 and num_b_in_batch <= len(non_b_indices):
                selected_indices = random.sample(non_b_indices, num_b_in_batch)
                list_c.extend(selected_indices)
        s_r_train_dl.dataset.TGA_backdoor_indices = list_c

    else:
        s_r_train_dl = None

    return train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
           backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl, s_r_train_dl, sr_ba_backdoor_target_indices, \
           labeled_indices_dict, backdoor_y_test_true
