from torchtext.data import Field, TabularDataset


class DataContainer:
    """A simple data container that utilises the Torchtext package, saving the
    train, valid and test data in TabularDataset
    """

    def __init__(self, train_data_path, valid_data_path, test_data_path, field_names=None):
        """
        Args:
            train_data_path:
            valid_data_path:
            test_data_path:
            field_names:
        """
        if field_names is None:
            # default field names are used for English-to-Chinese Transliteration
            field_names = ["en", "ch", "pinyin_str", "pinyin_char"]
        self.field_names = field_names
        # important that the order of tsv's columns is the same as in field_names
        self.fields = []
        print("Loading the Dataset into the container...")
        for name in field_names:
            self.fields.append((name, self.create_field()))  # modify this line if more requirement on Field
        self.dataset = {
            "train": self.create_dataset(train_data_path, self.fields),
            "valid": self.create_dataset(valid_data_path, self.fields),
            "test": self.create_dataset(test_data_path, self.fields)
        }
        print("Field names: {}".format(self.field_names))
        print("Data sizes: [(Train, {}), (Valid, {}), (Test, {})]".format(self.size(self.dataset["train"]),
                                                                          self.size(self.dataset["valid"]),
                                                                          self.size(self.dataset["test"])))
        self.show_train_examples()

    @staticmethod
    def tokenize(word):
        """The default tokenization method used for the tsv data

        Args:
            word:
        """
        word = word.replace('\n', '')
        return word.split(' ')

    @staticmethod
    def size(data_split):
        """Simple function that computes the size of the data split

        Args:
            data_split:
        """
        return len(data_split.examples)

    @classmethod
    def create_field(cls, tokenize=None, lower=True):
        """
        Args:
            tokenize:
            lower:
        """
        if not tokenize:
            tokenize = cls.tokenize  # the default tokenization method
        return Field(tokenize=tokenize,
                     init_token='<sos>',
                     eos_token='<eos>',
                     lower=lower,
                     include_lengths=True)

    @classmethod
    def create_dataset(cls, data_path, fields):
        """
        Args:
            data_path:
            fields:
        """
        return TabularDataset(path=data_path,
                              format='tsv',
                              skip_header=True,
                              fields=fields)

    def show_train_examples(self):
        """Present the first example for each data split"""
        print("The first example of the training data is:")
        first_example = vars(self.dataset["train"].examples[0])
        for k, v in first_example.items():
            print(k, ": ", v)
