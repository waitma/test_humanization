from model.encoder.model import ByteNetLMTime, AntiTFNet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from dataset.oas_unpair_dataset_new import OasUnPairDataset
from dataset.oas_pair_dataset_new import OasPairDataset
from torch.utils.data import Dataset
from torch.utils.data import Subset

import torch


def model_selected(config):
    if config.name == 'evo_oadm':
        return ByteNetLMTime(**config.model)
    elif config.name == 'trans_oadm':
        return AntiTFNet(**config.model)
    else:
        pass


def optimizer_selected(config, model):
    if config.train.optimizer.type == 'Adam':
        return Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.train.optimizer.lr,
            weight_decay=config.train.optimizer.weight_decay
        )
    elif config.train.optimizer.type == 'AdamW':
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.train.optimizer.lr,
            weight_decay=config.train.optimizer.weight_decay
        )
    else:
        pass


def scheduler_selected(config, optimizer):
    if config.train.scheduler.type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            factor=config.train.scheduler.factor,
            patience=config.train.scheduler.patience,
            min_lr=config.train.scheduler.min_lr
        )
    elif config.train.scheduler.type == 'cosine_annal':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.train.scheduler.T_max,
        )
    else:
        pass

def split_data(path, dataset):
    split = torch.load(path)
    subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
    return subsets

def get_dataset(root, name, version, split=True):
    if name == 'pair':
        dataset = OasPairDataset(root, version=version)
        split_path = dataset.index_path
        if split:
            return split_data(split_path, dataset)
        else:
            return dataset

    elif name == 'unpair':
        h_dataset = OasUnPairDataset(data_dpath=root, chaintype='heavy')
        l_dataset = OasUnPairDataset(data_dpath=root, chaintype='light')
        h_split_path = h_dataset.index_path
        l_split_path = l_dataset.index_path
        if split:
            h_subsets = split_data(h_split_path, h_dataset)
            l_subsets = split_data(l_split_path, l_dataset)
            return h_subsets, l_subsets
        else:
            return h_dataset, l_dataset

    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
