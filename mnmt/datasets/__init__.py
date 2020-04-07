from mnmt.inputter import DataContainer
import json
from pkg_resources import resource_filename

data_path = resource_filename("mnmt", "datasets")

# Import the DICT data_container
print("########### DICT DATASET ###########")
train = data_path + "/dict_data/train.tsv"
valid = data_path + "/dict_data/valid.tsv"
test = data_path + "/dict_data/test.tsv"
dict_data = DataContainer(train, valid, test)
DICT = {"name": 'DICT', "data_container": dict_data}
print("####################################")

# Import the NEWS data_container
print("########### NEWS DATASET ###########")
train = data_path + "/news_data/train.tsv"
valid = data_path + "/news_data/valid.tsv"
test = data_path + "/news_data/test.tsv"
news_data = DataContainer(train, valid, test)
# The test set contains multiple answers
with open(data_path + '/news_data/test.json', 'r') as f:
    test_json = json.load(f)
print("Load extra json file for the test set which contains multiple references.")
NEWS = {"name": 'NEWS',
        "data_container": news_data,
        "test-set-dict": test_json}
print("####################################")

__all__ = ["DICT", "NEWS", "data_path"]