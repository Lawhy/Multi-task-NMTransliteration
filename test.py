from mnmt.datasets import *
from mnmt.inputter import generate_batch_iterators
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def test_datasets():
    return generate_batch_iterators(DICT['dataset'], 64, device)


if __name__ == "__main__":
    print(test_datasets())
