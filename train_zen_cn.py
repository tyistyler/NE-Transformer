from models.TENER import TENER
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler, SequentialSampler
from fastNLP.embeddings import StaticEmbedding, BertEmbedding, StackEmbedding
from modules.pipe import CNNERPipe

from get_context import get_neighbor_for_vocab, build_instances

from run_token_level_classification import BertTokenizer, ZenNgramDict, MSRNgramDict, ZenForTokenClassification, load_examples
from utils_token_level_task import PeopledailyProcessor

import os
import argparse
from modules.callbacks import EvaluateCallback
from fastNLP import EarlyStopCallback
from datetime import datetime
from words_features import add_words_field_2_databundle

import torch
import random
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--context_num', type=int, default=10)
parser.add_argument('--glove_path', type=str)
parser.add_argument('--bert_model', type=str, required=True)
parser.add_argument('--zen_model', type=str, default="")
parser.add_argument('--pool_method', type=str, default="first", choices=["first", "last", "avg", "max"])
parser.add_argument('--trans_dropout', type=float, default=0.2)
parser.add_argument('--fc_dropout', type=float, default=0.4)
parser.add_argument('--memory_dropout', type=float, default=0.2)

# add
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dims', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument("--use_word_emb", action='store_true', help="Whether to add word_emb.")
parser.add_argument("--use_ngram", action='store_true', help="Whether to add ngram.")
parser.add_argument("--use_zen", action='store_true', help="Whether to add zen.")
parser.add_argument('--ngram_bert_dropout', type=float, default=0.2)
parser.add_argument('--convert_dropout', type=float, default=0.7)
# add --end
parser.add_argument('--fusion_type', type=str, default='gate-concat',
                    choices=['no-concat', 'concat', 'add', 'concat-add', 'gate-add', 'gate-concat'])
parser.add_argument('--fusion_dropout', type=float, default=0.2)
parser.add_argument('--highway_layer', type=int, default=0)
parser.add_argument('--kv_attn_type', type=str, default='dot')

args = parser.parse_args()


dataset = args.dataset

n_heads = args.n_heads
head_dims = args.head_dims
num_layers = args.num_layers
lr = args.lr
attn_type = 'adatrans'
n_epochs = 50


pos_embed = None

batch_size = args.batch_size
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True

fc_dropout=args.fc_dropout

# new_add
context_num = args.context_num
glove_path = args.glove_path
knowledge = True
feature_level = "all"
memory_dropout = args.memory_dropout
fusion_dropout = args.fusion_dropout
trans_dropout = args.trans_dropout
ngram_bert_dropout = args.ngram_bert_dropout
convert_dropout = args.convert_dropout
kv_attn_type = "dot"
fusion_type = "gate-concat"
highway_layer = 0
threshold = args.context_num

encoding_type = 'bioes'
name = 'caches/{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, normalize_embed, context_num)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)

def print_time():
    now = datetime.now()
    return "-".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])

save_path = "ckpt/{}_{}_{}.pth".format(dataset, model_type, context_num)

logPath = args.log

def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)



@cache_results(name, _refresh=False)
def load_data():

    paths = {'train': 'data/{}/train.txt'.format(dataset),
             'dev':'data/{}/dev.txt'.format(dataset),
             'test':'data/{}/test.txt'.format(dataset)}
    min_freq = 2
    data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(paths)

    dict_save_path = os.path.join("data/{}/data.pth".format(dataset))
    context_dict, context_word2id, context_id2word = get_neighbor_for_vocab(
        data_bundle.get_vocab('chars').word2idx, glove_path, dict_save_path
    )

    train_feature_data, dev_feature_data, test_feature_data = build_instances(
        "data/{}".format(dataset), context_num, context_dict
    )

    embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                            model_dir_or_name='data/gigaword_chn.all.a2b.uni.ite50.vec',
                            min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    bi_embed = StaticEmbedding(data_bundle.get_vocab('bigrams'),
                               model_dir_or_name='data/gigaword_chn.all.a2b.bi.ite50.vec',
                               word_dropout=0.02, dropout=0.3, min_freq=min_freq,
                               only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    tencent_embed = StaticEmbedding(data_bundle.get_vocab('chars'),
                                    model_dir_or_name='data/tencent_unigram.txt',
                                    min_freq=min_freq, only_norm_found_vector=normalize_embed, word_dropout=0.01,
                                    dropout=0.3)

    bert_embed = BertEmbedding(vocab=data_bundle.get_vocab('chars'), model_dir_or_name=args.bert_model, layers='-1',
                               pool_method=args.pool_method, word_dropout=0, dropout=0.5, include_cls_sep=False,
                               pooled_cls=True, requires_grad=False, auto_truncate=False)
    tencent_words_embed = None
    if args.use_word_emb:
        data_bundle = add_words_field_2_databundle(data_bundle)
        tencent_words_embed = StaticEmbedding(data_bundle.get_vocab('words'),
                                              model_dir_or_name='data/tencent_words_emb_for_weibo.txt',
                                              min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01,
                                              dropout=0.5)
        embed = StackEmbedding([embed, bert_embed], dropout=0, word_dropout=0.02)
    else:
        # embed = StackEmbedding([embed, tencent_embed, bert_embed], dropout=0, word_dropout=0.02)
        embed = StackEmbedding([embed, bert_embed], dropout=0, word_dropout=0.02)

    return data_bundle, embed, bi_embed, tencent_words_embed, train_feature_data, dev_feature_data, test_feature_data, context_word2id, context_id2word

data_bundle, embed, bi_embed, tencent_words_embed, train_feature_data, dev_feature_data, test_feature_data, feature2id, id2feature = load_data()

# data_bundle.get_dataset("train").print_field_meta()

vocab_size = len(data_bundle.get_vocab('chars'))
feature_vocab_size = len(feature2id)

# ZEN part
zen_model = None
zen_train_dataset = None
zen_dev_dataset = None
zen_test_dataset = None
if args.use_zen:
    print("[Info] Use ZEN !!! ")
    zen_model_path = args.zen_model
    processor = PeopledailyProcessor(dataset=dataset)
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(zen_model_path, do_lower_case=False)
    ngram_dict = ZenNgramDict(zen_model_path, tokenizer=tokenizer)
    zen_model = ZenForTokenClassification.from_pretrained(zen_model_path,
                                                      cache_dir="caches/",
                                                      num_labels=len(label_list),
                                                      multift=False)
    zen_model = zen_model.bert
    zen_model.to(device)
    zen_model.eval()
    data_dir = os.path.join("data", dataset)
    max_seq_len = 512

    zen_train_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="train")
    zen_dev_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="dev")
    zen_test_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="test")

    print("[Info] Zen Mode, Zen dataset loaded ...")

ngram_train_dataset = None
ngram_dev_dataset = None
ngram_test_dataset = None
if args.use_ngram:
    print("[Info] Use Ngram !!! ")
    vocab_path = 'memory_lexicon'
    zen_model_path = args.zen_model
    processor = PeopledailyProcessor(dataset=dataset)
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(zen_model_path, do_lower_case=False)
    ngram_dict = MSRNgramDict(vocab_path, tokenizer=tokenizer)
    data_dir = os.path.join("data", dataset)
    max_seq_len = 512

    ngram_train_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="train")
    ngram_dev_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="dev")
    ngram_test_dataset = load_examples(data_dir, max_seq_len, tokenizer, ngram_dict, processor, label_list, mode="test")

model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
              d_model=d_model, n_head=n_heads,
              feedforward_dim=dim_feedforward, dropout=trans_dropout,
              after_norm=after_norm, attn_type=attn_type,
              bi_embed=bi_embed, word_embed=tencent_words_embed,
              fc_dropout=fc_dropout,
              pos_embed=pos_embed,
              scale=attn_type=='naive',
              vocab_size=vocab_size,
              feature_vocab_size=feature_vocab_size,
              kv_attn_type=kv_attn_type,
              memory_dropout=memory_dropout,
              fusion_dropout=fusion_dropout, ngram_bert_dropout=ngram_bert_dropout, convert_dropout=convert_dropout,
              fusion_type=fusion_type,
              highway_layer=highway_layer,
              use_zen=args.use_zen, use_ngram=args.use_ngram
              )

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data=data_bundle.get_dataset('test'),
                                     test_feature_data=test_feature_data,
                                     feature2id=feature2id,
                                     id2feature=id2feature,
                                     context_num=context_num,
                                     use_zen=args.use_zen, use_ngram=args.use_ngram,
                                     zen_model=zen_model,
                                     ngram_test_dataset=ngram_test_dataset
                                     )

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=SequentialSampler(),
                  num_workers=0, n_epochs=n_epochs, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=save_path,
                  use_knowledge=False,
                  train_feature_data=train_feature_data,
                  test_feature_data=dev_feature_data,
                  feature2id=feature2id,
                  id2feature=id2feature,
                  logger_func=write_log,
                  context_num=context_num,
                  use_zen=args.use_zen, use_ngram=args.use_ngram,
                  zen_model=zen_model,
                  ngram_train_dataset=ngram_train_dataset,
                  ngram_dev_dataset=ngram_dev_dataset
                  )

trainer.train(load_best_model=False)
