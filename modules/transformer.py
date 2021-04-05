

import torch
import torch.nn.functional as F

from torch import nn
import math
from copy import deepcopy

from .relative_transformer import RelativeMultiHeadAttn

import sys
import numpy as np
import copy


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        """

        :param d_model:
        :param n_head:
        :param scale: 是否scale输出
        """
        super().__init__()
        assert d_model%n_head==0

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, mask):
        """

        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)

        return v


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
        """

        :param int d_model: 一般512之类的
        :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
            batch_size x max_len x d_model
        :param int feedforward_dim: FFN中间层的dimension的大小
        :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
        :param float dropout: 一共三个位置的dropout的大小
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.after_norm = after_norm

        self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(feedforward_dim, d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len x hidden_size
        :param mask: batch_size x max_len, 为0的地方为pad
        :return: batch_size x max_len x hidden_size
        """
        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x = self.self_attn(x, mask)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)
        residual = x
        if not self.after_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        if self.after_norm:
            x = self.norm2(x)
        return x

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, attention_probs_dropout_prob=0.3, num_attention_heads=4, output_attentions=False, keep_multihead_output=False):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # print("attention_scores:    ",attention_scores.size())
        # print("attention_mask:    ", attention_mask.size())
        # add
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # -------------------------------------------------------------
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.output_attentions:
            return attention_probs, context_layer
        return context_layer

# ok
class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.3, layer_norm_eps=1e-12):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ok
class BertAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=4, output_attentions=False, keep_multihead_output=False):
        super(BertAttention, self).__init__()
        self.output_attentions = output_attentions
        self.self = BertSelfAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.output = BertSelfOutput(hidden_size=hidden_size)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_output = self.self(input_tensor, attention_mask, head_mask)
        if self.output_attentions:
            attentions, self_output = self_output
        attention_output = self.output(self_output, input_tensor)
        if self.output_attentions:
            return attentions, attention_output
        return attention_output

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

# ok
class BertIntermediate(nn.Module):
    def __init__(self, intermediate_size=3072, hidden_size=768, hidden_act="gelu"):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str) or (sys.version_info[0] == 2 and isinstance(hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# ok
class BertOutput(nn.Module):
    def __init__(self, intermediate_size=3072, hidden_size=768, dropout_rate=0.5, layer_norm_eps=1e-12):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=4, dropout_rate=0.5, output_attentions=False, keep_multihead_output=False):
        super(BertLayer, self).__init__()
        self.output_attentions = output_attentions
        self.attention = BertAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads, output_attentions=output_attentions,
                                       keep_multihead_output=keep_multihead_output)
        self.intermediate = BertIntermediate(hidden_size=hidden_size)
        self.output = BertOutput(hidden_size=hidden_size, dropout_rate=dropout_rate)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        if self.output_attentions:
            attentions, attention_output = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if self.output_attentions:
            return attentions, layer_output
        return layer_output


class AdaptedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, ngram_embed_size=768, after_norm=True, attn_type='naive',
                 scale=False, dropout_attn=None, pos_embed=None, bert_dropout=0.5, convert_dropout=0.7):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                       for _ in range(num_layers)])

        self_attn_ngram = MultiHeadAttn(ngram_embed_size, n_head, dropout_attn, scale=scale)
        self.word_layers = nn.ModuleList([TransformerLayer(ngram_embed_size, deepcopy(self_attn_ngram), feedforward_dim, after_norm, dropout) for _ in range(1)])

        layer = BertLayer(hidden_size=ngram_embed_size, dropout_rate=bert_dropout, output_attentions=None, keep_multihead_output=None)
        self.num_hidden_ngram_layers = 1
        self.word_layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_ngram_layers)])
        self.convert = nn.Linear(d_model*2, d_model)
        self.convert_dropout = nn.Dropout(convert_dropout)

    def forward(self, x, mask, ngram_hidden_states=None, ngram_attention_mask=None, ngram_position_matrix=None):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)

            if i < self.num_hidden_ngram_layers:
                ngram_hidden_states = self.word_layers[i](ngram_hidden_states, ngram_attention_mask, None)

            # word_hidden = self.word_layers[i](ngram, ngram_mask)
            w_hidden = torch.bmm(ngram_position_matrix.float(), ngram_hidden_states.float())
            w_hidden = self.convert_dropout(w_hidden)
            # cat
            x = torch.cat([x, w_hidden], dim=-1)
            x = self.convert(x)
        return x




# class MultiHeadAttn(nn.Module):
#     def __init__(self, d_model, n_head, dropout=0.1, scale=False):
#         """
#
#         :param d_model:
#         :param n_head:
#         :param scale: 是否scale输出
#         """
#         super().__init__()
#         assert d_model%n_head==0
#
#         self.n_head = n_head
#         self.qkv_linear = nn.Linear(d_model, 3*d_model)
#         self.fc = nn.Linear(d_model, d_model)
#         self.dropout_layer = nn.Dropout(dropout)
#
#         if scale:
#             self.scale = math.sqrt(d_model//n_head)
#         else:
#             self.scale = 1
#
#     def forward(self, x, mask):
#         """
#
#         :param x: bsz x max_len x d_model
#         :param mask: bsz x max_len
#         :return:
#         """
#         batch_size, max_len, d_model = x.size()
#         x = self.qkv_linear(x)
#         q, k, v = torch.chunk(x, 3, dim=-1)
#         q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
#         k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
#         v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
#
#         attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
#         attn = attn/self.scale
#         attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))
#
#         attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
#         attn = self.dropout_layer(attn)
#         v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
#         v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
#         v = self.fc(v)
#
#         return v


# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, self_attn, feedforward_dim, after_norm, dropout):
#         """
#
#         :param int d_model: 一般512之类的
#         :param self_attn: self attention模块，输入为x:batch_size x max_len x d_model, mask:batch_size x max_len, 输出为
#             batch_size x max_len x d_model
#         :param int feedforward_dim: FFN中间层的dimension的大小
#         :param bool after_norm: norm的位置不一样，如果为False，则embedding可以直接连到输出
#         :param float dropout: 一共三个位置的dropout的大小
#         """
#         super().__init__()
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         self.self_attn = self_attn
#
#         self.after_norm = after_norm
#
#         self.ffn = nn.Sequential(nn.Linear(d_model, feedforward_dim),
#                                  nn.ReLU(),
#                                  nn.Dropout(dropout),
#                                  nn.Linear(feedforward_dim, d_model),
#                                  nn.Dropout(dropout))
#
#     def forward(self, x, mask):
#         """
#
#         :param x: batch_size x max_len x hidden_size
#         :param mask: batch_size x max_len, 为0的地方为pad
#         :return: batch_size x max_len x hidden_size
#         """
#         residual = x
#         if not self.after_norm:
#             x = self.norm1(x)
#
#         x = self.self_attn(x, mask)
#         x = x + residual
#         if self.after_norm:
#             x = self.norm1(x)
#         residual = x
#         if not self.after_norm:
#             x = self.norm2(x)
#         x = self.ffn(x)
#         x = residual + x
#         if self.after_norm:
#             x = self.norm2(x)
#         return x



class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, feedforward_dim, dropout, after_norm=True, attn_type='naive',
                 scale=False, dropout_attn=None, pos_embed=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, 0, init_size=1024)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(1024, d_model, 0)

        if attn_type == 'transformer':
            self_attn = MultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)
        elif attn_type == 'adatrans':
            self_attn = RelativeMultiHeadAttn(d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(d_model, deepcopy(self_attn), feedforward_dim, after_norm, dropout)
                       for _ in range(num_layers)])

    def forward(self, x, mask):
        """

        :param x: batch_size x max_len
        :param mask: batch_size x max_len. 有value的地方为1
        :return:
        """
        if self.pos_embed is not None:
            x = x + self.pos_embed(mask)

        for layer in self.layers:
            x = layer(x, mask)
        return x


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        # positions: batch_size x max_len, 把words的index输入就好了
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)