# English-to-Chinese Transliteration with a Phonetic Auxiliary Task

This repository contains the Pytorch-based implementation of the Multi-task Neural Machine Transliteration system submitted to
[The 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing](http://aacl2020.org/). 

It can be accessed in Yuan He and Shay B. Cohen: *[English-to-Chinese Transliteration with a Phonetic Auxiliary Task](https://www.aclweb.org/anthology/2020.aacl-main.40/)*, AACL-IJCNLP, 2020.


Citation:
```
@inproceedings{he-cohen-2020-english,
    title = "{E}nglish-to-{C}hinese Transliteration with Phonetic Auxiliary Task",
    author = "He, Yuan  and
      Cohen, Shay B.",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.aacl-main.40",
    pages = "378--388",
    abstract = "Approaching named entities transliteration as a Neural Machine Translation (NMT) problem is common practice. While many have applied various NMT techniques to enhance machine transliteration models, few focus on the linguistic features particular to the relevant languages. In this paper, we investigate the effect of incorporating phonetic features for English-to-Chinese transliteration under the multi-task learning (MTL) setting{---}where we define a phonetic auxiliary task aimed to improve the generalization performance of the main transliteration task. In addition to our system, we also release a new English-to-Chinese dataset and propose a novel evaluation metric which considers multiple possible transliterations given a source name. Our results show that the multi-task model achieves similar performance as the previous state of the art with a model of a much smaller size.",
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
