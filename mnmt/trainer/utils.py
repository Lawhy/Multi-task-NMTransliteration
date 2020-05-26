from mnmt.inputter import DataContainer
import numpy as np
import random
import torch
import torch.nn as nn


def set_reproducibility(seed=1234):
    """
    Args:
        seed: number of seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vocabs(data_container: DataContainer, dict_min_freqs: dict):
    """
    Args:
        data_container (DataContainer):
        dict_min_freqs (dict): minimum frequencies are thresholds for the vocab building
    """
    assert len(data_container.fields) == len(dict_min_freqs)
    train_data = data_container.dataset["train"]
    for (name, field) in data_container.fields:
        field.build_vocab(train_data, min_freq=dict_min_freqs[name])


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
    # model.apply(init_weights)


def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')
    return num_params


def create_mask(src, src_pad_idx):
    mask = (src != src_pad_idx).permute(1, 0)
    return mask


def log_print(log_path, statement):
    print(statement)
    with open(log_path, 'a+') as f:
        f.write(f'{statement}\n')


def inflate(tensor, times, dim):
    """
    Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
    Args:
        tensor: A :class:`Tensor` to inflate
        times: number of repetitions
        dim: axis for inflation (default=0)
    Returns:
        A :class:`Tensor`
    Examples::
        >> a = torch.LongTensor([[1, 2], [3, 4]])
        >> a
        1   2
        3   4
        [torch.LongTensor of size 2x2]
        >> b = ._inflate(a, 2, dim=1)
        >> b
        1   2   1   2
        3   4   3   4
        [torch.LongTensor of size 2x4]
        >> c = _inflate(a, 2, dim=0)
        >> c
        1   2
        3   4
        1   2
        3   4
        [torch.LongTensor of size 4x2]
    """
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)
