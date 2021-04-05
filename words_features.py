# coding: utf-8
# Name:     test
# Author:   dell
# Data:     2021/3/5
from utils_token_level_task import PeopledailyProcessor
import os

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from modules.pipe import CNNERPipe
import torch

def _read_txt(src):
    # 一行一个词语
    f = open(src, 'r', encoding='utf-8')
    lines = f.readlines()
    seqs = []
    seq = []
    for line in lines:
        if line != '\n':
            word, _ = line.strip('\ufeff\n').split()
            seq.append(word)
        else:
            seqs.append(seq)
            seq = []
    return seqs

def read_txt_ont_dim(src):
    # 一行一个词语
    f = open(src, 'r', encoding='utf-8')
    lines = f.readlines()
    seqs = []
    # seq = []
    for line in lines:
        if line != '\n':
            word, _ = line.strip('\ufeff\n').split()
            seqs.append(word)
        else:
            continue
    return seqs

def get_corpus_words(train, dev, test):
    train_list = read_txt_ont_dim(train)
    dev_list = read_txt_ont_dim(dev)
    test_list = read_txt_ont_dim(test)

    corpus = train_list + dev_list + test_list
    # print(corpus[0])
    return corpus


def add_words_field_2_databundle(data_bundle):
    train_cws_field = "data/wb_cws/train_cws_word.txt"
    dev_cws_field = "data/wb_cws/dev_cws_word.txt"
    test_cws_field = "data/wb_cws/test_cws_word.txt"

    train_field = _read_txt(train_cws_field)
    dev_field = _read_txt(dev_cws_field)
    test_field = _read_txt(test_cws_field)
    #
    #
    data_bundle.get_dataset('train').add_field(field_name="raw_words", fields=train_field)
    data_bundle.get_dataset('dev').add_field(field_name="raw_words", fields=dev_field)
    data_bundle.get_dataset('test').add_field(field_name="raw_words", fields=test_field)

    # 添加词表
    words_vocab = Vocabulary()
    word_list = get_corpus_words(train_cws_field, dev_cws_field, test_cws_field)
    words_vocab.update(word_list)
    data_bundle.set_vocab(words_vocab, field_name="words")

    # 将raw_words转换为words_id
    for dataset in ["train", "dev", "test"]:
        raw_words = list(data_bundle.get_dataset(dataset)["raw_words"])
        words_ids = []
        for words in raw_words:
            words_id = []
            for word in words:
                words_id.append(words_vocab.to_index(word))
            words_ids.append(words_id)
        data_bundle.get_dataset(dataset).add_field(field_name="words", fields=words_ids)
    data_bundle.set_input('words')
    data_bundle.set_ignore_type('words', flag=False)
    data_bundle.set_pad_val("words", 0)
    return data_bundle


# if __name__ == '__main__':
#
#     dataset = "weibo"
#     encoding_type = "bioes"
#     paths = {'train': 'data/{}/train.txt'.format(dataset),
#                  'dev':'data/{}/dev.txt'.format(dataset),
#                  'test':'data/{}/test.txt'.format(dataset)}
#     min_freq = 2
#     data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)
#     #
#     train_cws_field = "data/wb_cws/train_cws_word.txt"
#     dev_cws_field = "data/wb_cws/dev_cws_word.txt"
#     test_cws_field = "data/wb_cws/test_cws_word.txt"
#
#     train_field = _read_txt(train_cws_field)
#     dev_field = _read_txt(dev_cws_field)
#     test_field = _read_txt(test_cws_field)
#     #
#     #
#     data_bundle.get_dataset('train').add_field(field_name="raw_words", fields=train_field)
#     data_bundle.get_dataset('dev').add_field(field_name="raw_words", fields=dev_field)
#     data_bundle.get_dataset('test').add_field(field_name="raw_words", fields=test_field)
#
#     # 添加词表
#     words_vocab = Vocabulary()
#     word_list = get_corpus_words(train_cws_field, dev_cws_field, test_cws_field)
#     words_vocab.update(word_list)
#     data_bundle.set_vocab(words_vocab, field_name="words")
#
#     # 将raw_words转换为words_id
#     for dataset in ["train", "dev", "test"]:
#         raw_words = list(data_bundle.get_dataset(dataset)["raw_words"])
#         words_ids = []
#         for words in raw_words:
#             words_id = []
#             for word in words:
#                 words_id.append(words_vocab.to_index(word))
#             words_ids.append(words_id)
#         data_bundle.get_dataset(dataset).add_field(field_name="words", fields=words_ids)
#
#     print(data_bundle.get_dataset("test"))
#     # print(words_vocab.to_index("科技"))
#
#     normalize_embed = False
#
#     tencent_embed = StaticEmbedding(data_bundle.get_vocab('words'),
#                                     model_dir_or_name='data/tencent_words.txt',
#                                     min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0,
#                                     dropout=0)
#
#     # print(tencent_embed(torch.tensor(30)))

