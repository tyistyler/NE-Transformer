import os

# dataset name
dataset = "weibo"
seed = 66
attn_type = "dot"
fusion_type = "gate-concat"
# Path of bert model
bert_model = "data/bert-base-chinese"
# Path of the pre-trained word embeddings for getting similar words for each token
glove_path = "data/tencent_unigram.txt"
# Path of the ZEN model
zen_model = "zen_base/"

log = "log/bert_{}.txt".format(dataset)
# batch_size = 32
# num_layers = 1 lr = 0.0001
os.system("python3 train_zen_cn.py --dataset {} "
          "--batch_size 32 --num_layers 1 --n_heads 4 --head_dims 64 "
          "--seed 66 --kv_attn_type {} --fusion_type {} --context_num 10 "
          "--bert_model {} --pool_method first --glove_path {} --zen_model data/ZEN_pretrain_base "
          "--lr 0.0001 --trans_dropout 0.3 --fc_dropout 0.4 --memory_dropout 0.4 "
          "--fusion_dropout 0.4 --log {} --use_word_emb --use_ngram --ngram_bert_dropout 0.3 --convert_dropout 0.7 ".format(dataset, attn_type, fusion_type, bert_model, glove_path, log))
