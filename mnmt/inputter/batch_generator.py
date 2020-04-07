from torchtext.data import BucketIterator


def generate_batch_iterators(data_container, batch_size, device, src_lang):
    """
    Args:
        data_container (DataContainer): object that handles the train, valid and test data
        batch_size (int):
        device (torch.device): GPU or CPU
        src_lang: name of the source language
    """
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (data_container.dataset["train"], data_container.dataset["valid"], data_container.dataset["test"]),
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(getattr(x, src_lang)),
        device=device)
    return train_iterator, valid_iterator, test_iterator
