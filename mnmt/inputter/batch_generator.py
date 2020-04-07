from torchtext.data import BucketIterator


def generate_batch_iterators(data_container, batch_size, device):
    """
    Args:
        data_container (DataContainer): object that handles the train, valid and test data
        batch_size (int):
        device (torch.device): GPU or CPU
    """
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (data_container.dataset["train"], data_container.dataset["valid"], data_container.dataset["test"]),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)
    return train_iterator, valid_iterator, test_iterator
