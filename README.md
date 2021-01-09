# English-to-Chinese Transliteration with a Phonetic Auxiliary Task

This repository contains the Pytorch-based implementation of the Multi-task Neural Machine Transliteration system submitted to
[The 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing](http://aacl2020.org/). 

It can be accessed in Yuan He and Shay B. Cohen: *[English-to-Chinese Transliteration with a Phonetic Auxiliary Task](https://www.aclweb.org/anthology/2020.aacl-main.40/)*, AACL-IJCNLP, 2020.


Citation:
```
@inproceedings{He2020EnglishtoChineseTW,
  title={English-to-Chinese Transliteration with Phonetic Auxiliary Task},
  author={Yuan He and Shay B. Cohen},
  booktitle={AACL/IJCNLP},
  year={2020}
}
```

------------
## Datasets
This paper involves two English-to-Chinese transliteration datasets: 

   - NEWS dataset, originally taken from [NEWS 2018](http://workshop.colips.org/news2018/dataset.html);

   - DICT dataset, a new dataset we released.
  
The datasets in `.tsv` format are available at `mnmt/datasets`.

## Model
The essential components of our model are implemented individually and the code is available in the directory `mnmt`,
e.g. the encoder is available at `mnmt/encoder/basic_encoder.py`. See below for the quick access of our model:

1. Requirements
```
torchtext>=0.5.0
torchvision>=0.5.0
python-Levenshtein>=0.12.0  # for Minimum Edit Distance
pandas>=1.0.3
```
2. Install package
```
pip install multi-nmt-lawhy
```
OR simply clone this git repository.

3. Training Script
```
# Example training script for NEWS dataset.
python Seq2MultiSeq.py
```
