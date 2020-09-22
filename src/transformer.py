"""
Copyright 2020 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""Transformer Seq2Seq model with COPY.

Reference:
https://www.tensorflow.org/tutorials/text/transformer
"""
import math
import numpy as np

import tensorflow as tf


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(
      np.arange(position)[:, np.newaxis],
      np.arange(d_model)[np.newaxis, :], d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask, dropout, training=False):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k).
      Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits,
                                    axis=-1)  # (..., seq_len_q, seq_len_k)
  #attention dropout. commented out due to perf decrease
  attention_weights = dropout(attention_weights, training=training)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


def create_padding_mask(seq, pad_id):
  seq = tf.cast(tf.math.equal(seq, pad_id), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def create_masks(src_token_idx, tgt_token_ids, src_pad_id, tgt_pad_id):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(src_token_idx, src_pad_id)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(src_token_idx, src_pad_id)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tgt_token_ids)[1])
  dec_target_padding_mask = create_padding_mask(tgt_token_ids, tgt_pad_id)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, rate=0.1):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

    self.dropout = tf.keras.layers.Dropout(rate)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).

    Transpose the result such that the shape is (batch_size, num_heads, seq_len,
    depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask, training=False, mem=None):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(
        q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(
        k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(
        v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    if mem is not None:
      if "k" in mem and "v" in mem:
        mem["k"] = tf.concat([mem["k"], k], axis=2)
        mem["v"] = tf.concat([mem["v"], k], axis=2)
        k = mem["k"]
        v = mem["v"]
      else:
        mem["k"] = k
        mem["v"] = v

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask, self.dropout, training=training)

    # (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # (batch_size, seq_len_q, d_model)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # (batch_size, seq_len_q, d_model)
    output = self.dense(concat_attention)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  #gelu?
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff,
                            activation="relu"),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, rate)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(
        x, x, x, mask,
        training=training)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x +
                           attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 +
                           ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class Encoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               source_vocab_size,
               maximum_position_encoding,
               rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(source_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

    self.enc_layers = [
        EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
    ]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]

    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads, rate)
    self.mha2 = MultiHeadAttention(d_model, num_heads, rate)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self,
           x,
           enc_output,
           training,
           look_ahead_mask,
           padding_mask,
           step=-1,
           mem=None):
    # if step == -1: decode on whole x
    # else decode on x[:, step]
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    q = x

    # decode_len = tgt_seq_len if step = -1 else 1
    # (batch_size, decode_len, d_model)

    if mem is not None:
      #tf.print(x)
      #print(x)
      assert x.shape.as_list()[1] == 1
      attn1, attn_weights_block1 = self.mha1(mem,
                                             mem,
                                             q,
                                             None,
                                             training=training)
    else:
      attn1, attn_weights_block1 = self.mha1(x,
                                             x,
                                             q,
                                             look_ahead_mask,
                                             training=training)

    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + q)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask,
        training=training)  # (batch_size, decode_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, decode_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, decode_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output +
                           out2)  # (batch_size, decode_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
  """Transformer decoder with COPY."""

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               tgt_vocab,
               src_seq_len,
               maximum_position_encoding,
               rate=0.1,
               predict_edge=False):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.tgt_vocab = tgt_vocab
    self.tgt_vocab_size = len(tgt_vocab)
    self.src_seq_len = src_seq_len

    self.embedding = tf.keras.layers.Embedding(self.tgt_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [
        DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
    ]

    self.gen_scores_layer = tf.keras.layers.Dense(self.tgt_vocab_size)
    self.copy_scores_proj_layer = tf.keras.layers.Dense(self.d_model,
                                                        activation="tanh")

    self.dropout = tf.keras.layers.Dropout(rate)
    self.predict_edge = predict_edge
    if self.predict_edge:
      self.merge_embedding = tf.keras.layers.Dense(self.d_model,
                                                   activation="tanh")
      self.edge_q = tf.keras.layers.Dense(self.d_model)
      self.edge_k = tf.keras.layers.Dense(self.d_model)

  def _process_tgt_token_ids(self, tgt_token_ids):
    """Prepare decode_step input.

    Args:
      tgt_token_ids: <int64>[batch_sz, tgt_seq_len].

    Returns:
      A pair of:
        is_target_type: <bool>[batch_sz, tgt_seq_len]
        dec_input_src_selection: <int64>[batch_sz, tgt_seq_len, src_seq_len].
    """
    is_target_type = tgt_token_ids < self.tgt_vocab_size
    # Shape: [batch_sz, tgt_seq_len, src_seq_len]
    # Map the index back to starting from 0.
    # Note that, for each batch i, tgt token j where
    # is_target_type[i, j] = True, sum(dec_input_src_selection[i, j, *]) = 0,
    # i.e., no source token will be selected.
    dec_input_src_selection = tf.one_hot(
        (tgt_token_ids - self.tgt_vocab_size) *
        (1 - tf.cast(is_target_type, dtype=tgt_token_ids.dtype)),
        depth=self.src_seq_len)
    # Remove those 0 indices that are mistakenly flipped.
    dec_input_src_selection = dec_input_src_selection * (
        1 - tf.cast(tf.expand_dims(is_target_type, -1),
                    dtype=dec_input_src_selection.dtype))
    return is_target_type, dec_input_src_selection

  def _gen_scores(self, x):
    """Returns [batch_sz, tgt_seq_len, tgt_vocab_size]."""
    return self.gen_scores_layer(x)

  def _copy_scores(self, x, enc_output):
    """Returns [batch_sz, tgt_seq_len, src_seq_len]."""
    # Shape [batch_sz, src_seq_len, d_model].
    copy_proj = self.copy_scores_proj_layer(enc_output)
    # Shape = [batch_sz, 1, src_seq_len, d_model] x [batch_sz, tgt_seq_len, d_model, 1]
    # = [batch_sz, tgt_seq_len, src_seq_len, 1]
    copy_scores = tf.matmul(tf.expand_dims(copy_proj, 1), tf.expand_dims(x, -1))
    return tf.squeeze(copy_scores, -1)

  def _get_input_embedding(self, x, enc_output):
    # is_target_type.shape = [batch_sz, tgt_seq_len]
    # dec_input_src_selection.shape = [batch_sz, tgt_seq_len, src_seq_len]
    is_target_type, dec_input_src_selection = self._process_tgt_token_ids(x)
    # (batch_size, tgt_seq_len, d_model)
    x = self.embedding(x * tf.cast(is_target_type, dtype=x.dtype))
    # mask out those copied from src
    x *= tf.expand_dims(tf.cast(is_target_type, dtype=x.dtype), -1)
    # (batch_sz, tgt_seq_len, d_model)
    aggregated_src = tf.reduce_sum(
        # (batch_size, 1, seq_seq_len, d_model) * (batch_sz, tgt_seq_len, src_seq_len, 1)
        tf.expand_dims(enc_output, 1) *
        tf.expand_dims(dec_input_src_selection, -1),
        axis=2)
    x += aggregated_src
    return x

  def call(self,
           x,
           enc_output,
           training,
           look_ahead_mask,
           padding_mask,
           mem=None,
           tgt_edges=None):

    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    attention_weights = {}
    if tgt_edges is not None:
      if mem is not None:
        tgt_edges = tf.expand_dims(tgt_edges[:, -1], -1)
      mask = tf.cast(tf.math.equal(tgt_edges, -1), tf.int64)
      mask_reverse = tf.expand_dims(
          tf.cast(tf.math.not_equal(tgt_edges, -1), tf.float32), 2)
      tgt_edges += mask
      edge_x = tf.reshape(
          tf.gather_nd(tf.expand_dims(x, 2),
                       indices=tf.expand_dims(tgt_edges, 2),
                       batch_dims=1), [batch_size, -1])
      edge_x = self._get_input_embedding(edge_x, enc_output)
      edge_x *= mask_reverse
    # (batch_size, tgt_seq_len, d_model)
    if mem is not None:
      x = tf.expand_dims(x[:, -1], -1)

    x = self._get_input_embedding(x, enc_output)

    if tgt_edges is not None:
      x = self.merge_embedding(tf.concat([x, edge_x], axis=2))

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    if mem is not None:
      x += self.pos_encoding[:, (seq_len - 1):seq_len, :]
    else:
      x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    if mem is not None:
      assert x.shape.as_list()[1] == 1
      if 0 in mem:
        mem[0] = tf.concat([mem[0], x], axis=1)
      else:
        mem[0] = x

    for i in range(self.num_layers):
      # x.shape == (batch_size, tgt_seq_len, d_model)
      x, block1, block2 = self.dec_layers[i](
          x,
          enc_output,
          training,
          look_ahead_mask,
          padding_mask,
          mem=mem[i] if mem is not None else None)
      if mem is not None:
        assert x.shape.as_list()[1] == 1
        if i + 1 in mem:
          mem[i + 1] = tf.concat([mem[i + 1], x], axis=1)
        else:
          mem[i + 1] = x
      attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
      attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

    # TODO: should we consider attentions here?
    # (batch_size, tgt_seq_len, tgt_vocab_size)
    gen_scores = self._gen_scores(x)
    # (batch_sz, tgt_seq_len, src_seq_len)
    copy_scores = self._copy_scores(x, enc_output)

    scores = tf.concat([gen_scores, copy_scores], axis=2)

    if self.predict_edge:
      q = self.edge_q(x)
      if mem is not None:
        k = self.edge_k(mem[self.num_layers])
      else:
        k = self.edge_k(x)
      edge_scores = tf.matmul(q, k, transpose_b=True)
      if mem is None:
        edge_scores += (tf.squeeze(look_ahead_mask, 1) * -1e9)
      return scores, attention_weights, edge_scores
    else:
      return scores, attention_weights


class Transformer(tf.keras.Model):
  default_hparams = {
      "num_layers": 3,
      "d_model": 128,
      "num_heads": 8,
      "dff": 512,
      "num_src_tokens": None,
      "num_tgt_tokens": None,
      "dropout": 0.2,
  }

  def __init__(self, src_vocab, tgt_vocab, hparams=None):
    super(Transformer, self).__init__()
    self.hparams = dict(Transformer.default_hparams)
    if hparams:
      for k, v in hparams.items():
        if k in self.hparams:
          self.hparams[k] = v
    self.src_vocab = src_vocab
    self.src_vocab_size = len(src_vocab)
    self.tgt_vocab = tgt_vocab
    self.tgt_vocab_size = len(tgt_vocab)
    self.src_seq_len = self.hparams["num_src_tokens"]
    self.tgt_seq_len = self.hparams["num_tgt_tokens"]
    self.predict_edge = hparams["predict_edge"]

    self.encoder = Encoder(num_layers=self.hparams["num_layers"],
                           d_model=self.hparams["d_model"],
                           num_heads=self.hparams["num_heads"],
                           dff=self.hparams["dff"],
                           source_vocab_size=self.src_vocab_size,
                           maximum_position_encoding=self.src_seq_len,
                           rate=self.hparams["dropout"])

    self.decoder = Decoder(num_layers=self.hparams["num_layers"],
                           d_model=self.hparams["d_model"],
                           num_heads=self.hparams["num_heads"],
                           dff=self.hparams["dff"],
                           tgt_vocab=self.tgt_vocab,
                           src_seq_len=self.src_seq_len,
                           maximum_position_encoding=self.tgt_seq_len,
                           rate=self.hparams["dropout"],
                           predict_edge=self.predict_edge)

  def call(self, examples, beam_size=1, is_train=True, return_all=False):
    if is_train:
      return self.train(examples, is_train)
    else:
      #return self.greedy_decode2(examples, is_train)
      return self.beam(examples,
                       beam_size=beam_size,
                       is_train=is_train,
                       return_all=return_all)

  def train(self, examples, is_train=True):
    src_token_ids = examples["src_token_ids"]
    tgt_token_ids = examples["tgt_token_ids"]
    if self.predict_edge:
      tgt_edges = examples["tgt_edges"]

    # enc_padding_mask: (batch_size, 1, 1, src_seq_len)
    # combined_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
    # dec_padding_mask: (batch_size, 1, 1, src_seq_len)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        src_token_ids, tgt_token_ids,
        self.src_vocab.token2idx[self.src_vocab.PAD],
        self.tgt_vocab.token2idx[self.tgt_vocab.PAD])

    # (batch_size, src_seq_len, d_model)
    enc_output = self.encoder(src_token_ids, is_train, enc_padding_mask)

    # dec_output.shape == (batch_size, tgt_seq_len, tgt_vocab_size+src_seq_len)
    if self.predict_edge:
      dec_output, _, edge_scores = self.decoder(tgt_token_ids,
                                                enc_output,
                                                is_train,
                                                combined_mask,
                                                dec_padding_mask,
                                                tgt_edges=tgt_edges)
    else:
      dec_output, _ = self.decoder(tgt_token_ids, enc_output, is_train,
                                   combined_mask, dec_padding_mask)
    # prepend the BOS token
    # (batch_size, 1)
    start_token = tf.expand_dims(tgt_token_ids[:, 0], axis=-1)
    # (batch_size, 1, tgt_vocab_size + src_seq_len)
    start_token_onehot = tf.one_hot(start_token,
                                    depth=(self.tgt_vocab_size +
                                           self.src_seq_len))
    start_token_logits = start_token_onehot + (start_token_onehot - 1) * 1e9
    dec_output = tf.concat([start_token_logits, dec_output[:, :-1, :]], axis=1)

    # (batch_size, tgt_seq_len, tgt_vocab_size+src_seq_len)
    if self.predict_edge:
      return dec_output, edge_scores
    else:
      return dec_output

  def greedy_decode(self,
                    examples,
                    is_train=False,
                    tgt_seq_len=None,
                    fast=True):
    # at each step, decode with whole output prefix
    src_token_ids = examples["src_token_ids"]

    if not tgt_seq_len:
      tgt_seq_len = self.tgt_seq_len

    enc_padding_mask = create_padding_mask(
        src_token_ids, self.src_vocab.token2idx[self.src_vocab.PAD])
    dec_padding_mask = create_padding_mask(
        src_token_ids, self.src_vocab.token2idx[self.src_vocab.PAD])
    # (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(src_token_ids, is_train, enc_padding_mask)
    batch_size = tf.shape(enc_output)[0]
    start_token = tf.reshape(
        tf.cast(tf.repeat(self.tgt_vocab.token2idx[self.tgt_vocab.BOS],
                          repeats=batch_size),
                dtype=tf.int64), [-1, 1])
    tgt_inputs = start_token
    tgt_edges = tf.zeros([batch_size, 1], dtype=tf.int64) - 1

    start_token_onehot = tf.one_hot(start_token,
                                    depth=(self.tgt_vocab_size +
                                           self.src_seq_len))
    start_token_logits = start_token_onehot + (start_token_onehot - 1) * 1e9
    output = [start_token_logits]
    edge_output = []

    if fast:
      mem = {}
    else:
      mem = None

    for t in range(1, tgt_seq_len):
      look_ahead_mask = create_look_ahead_mask(t)[tf.newaxis, tf.newaxis, :, :]
      # dec_output.shape == (batch_sz, t, tgt_vocab_size+src_seq_len)
      if self.predict_edge:
        dec_output, _, edge_scores = self.decoder(tgt_inputs,
                                                  enc_output,
                                                  is_train,
                                                  look_ahead_mask,
                                                  dec_padding_mask,
                                                  mem=mem,
                                                  tgt_edges=tgt_edges)
      else:
        dec_output, _ = self.decoder(tgt_inputs,
                                     enc_output,
                                     is_train,
                                     look_ahead_mask,
                                     dec_padding_mask,
                                     mem=mem)
      # (batch_sz, tgt_vocab_size+src_seq_len)
      last_step_output = dec_output[:, -1, :]
      last_step_output_idx = tf.expand_dims(tf.argmax(last_step_output, axis=1),
                                            axis=-1)
      tgt_inputs = tf.concat([tgt_inputs, last_step_output_idx], axis=-1)

      if self.predict_edge:
        last_step_score = edge_scores[:, -1, :]
        last_step_score_idx = tf.expand_dims(tf.argmax(last_step_score, axis=1),
                                             axis=-1)
        tgt_edges = tf.concat([tgt_edges, last_step_score_idx], axis=-1)

      # (batch_sz, t+1)
      if fast:
        output.append(dec_output)
        if self.predict_edge:
          edge_output.append(
              tf.concat([
                  edge_scores,
                  tf.fill([batch_size, 1, tgt_seq_len - t], -1e9)
              ],
                        axis=2))

    # here dec_output.shape = (batch_sz, tgt_seq_len-1, tgt_vocab_size+src_seq_len)
    # prepend the BOS token
    # (batch_size, 1, tgt_vocab_size + src_seq_len)
    if not fast:
      output.append(dec_output)
      if self.predict_edge:
        edge_output.append(edge_scores)
    dec_output = tf.concat(output, axis=1)
    if self.predict_edge:
      edge_output.append(tf.fill([batch_size, 1, tgt_seq_len], -1e9))
      edge_output = tf.concat(edge_output, axis=1)
    if self.predict_edge:
      return dec_output, edge_output
    else:
      return dec_output

  @tf.function
  def beam(self,
           examples,
           beam_size=5,
           len_penalty=1,
           stop_early=True,
           normalize_scores=True,
           is_train=False,
           tgt_seq_len=None,
           return_all=False):

    if beam_size == 1:
      return self.greedy_decode(examples, is_train, tgt_seq_len)

    src_token_ids = examples["src_token_ids"]
    batch_size = len(src_token_ids)
    src_token_ids = tf.repeat(src_token_ids,
                              beam_size * tf.ones([batch_size], dtype=tf.int32),
                              axis=0)

    if not tgt_seq_len:
      tgt_seq_len = self.tgt_seq_len

    enc_padding_mask = create_padding_mask(
        src_token_ids, self.src_vocab.token2idx[self.src_vocab.PAD])
    dec_padding_mask = create_padding_mask(
        src_token_ids, self.src_vocab.token2idx[self.src_vocab.PAD])
    # (batch_size x beam_size, inp_seq_len, d_model)
    enc_output = self.encoder(src_token_ids, is_train, enc_padding_mask)

    start_token = tf.reshape(
        tf.cast(tf.repeat(self.tgt_vocab.token2idx[self.tgt_vocab.BOS],
                          repeats=batch_size * beam_size),
                dtype=tf.int32), [-1, 1])
    tgt_inputs = start_token

    scores = tf.zeros([batch_size * beam_size, 1])  # (bsz x beam) x T

    mem = {}
    reorder_state = None
    batch_idxs = None
    bbsz_offset = tf.expand_dims(tf.range(batch_size) * beam_size, 1)  # bsz x 1

    for t in range(1, tgt_seq_len):

      # reorder mem, enc_output and masks
      if reorder_state is not None:
        if batch_idxs is not None:
          a = batch_idxs - tf.expand_dims(tf.range(len(batch_idxs)),
                                          1)  #offset back
          reorder_state = tf.reshape(
              tf.reshape(reorder_state, [-1, beam_size]) + a * beam_size,
              [-1, 1])
        for i in range(len(mem)):
          mem[i] = tf.gather_nd(mem[i], indices=reorder_state)
        enc_output = tf.gather_nd(enc_output, indices=reorder_state)
        dec_padding_mask = tf.gather_nd(dec_padding_mask, indices=reorder_state)

      look_ahead_mask = create_look_ahead_mask(t)

      dec_output, _ = self.decoder(tgt_inputs,
                                   enc_output,
                                   is_train,
                                   look_ahead_mask,
                                   dec_padding_mask,
                                   mem=mem)
      probs = tf.nn.log_softmax(dec_output, axis=-1)[:,
                                                     -1, :]  # (bsz x beam) x V
      finalized_hypos_mask = tf.expand_dims(
          tf.cast(tf.math.equal(tgt_inputs[:, -1],
                                self.tgt_vocab.token2idx[self.tgt_vocab.EOS]),
                  dtype=tf.float32), 1)
      finalized_hypos_mask += tf.expand_dims(
          tf.cast(tf.math.equal(tgt_inputs[:, -1],
                                self.tgt_vocab.token2idx[self.tgt_vocab.PAD]),
                  dtype=tf.float32), 1)
      finalized_hypos_mask_reverse = tf.cast(tf.math.logical_not(
          tf.cast(finalized_hypos_mask, dtype=tf.bool)),
                                             dtype=tf.float32)

      if t != 1:
        probs += tf.tile(
            tf.expand_dims(
                tf.cast(tf.not_equal(
                    tf.range(probs.shape[1]),
                    self.tgt_vocab.token2idx[self.tgt_vocab.PAD]),
                        dtype=tf.float32) * -1e9, 0),
            tf.reshape(tf.constant([batch_size * beam_size, 1]),
                       [-1])) * finalized_hypos_mask
        temp = (1 - tf.tile(
            tf.expand_dims(
                tf.cast(tf.equal(tf.range(probs.shape[1]),
                                 self.tgt_vocab.token2idx[self.tgt_vocab.PAD]),
                        dtype=tf.float32), 0),
            tf.reshape(tf.constant([batch_size * beam_size, 1]), [-1])) *
                finalized_hypos_mask)
        probs *= temp
        probs += tf.tile(
            tf.expand_dims(
                tf.cast(tf.equal(tf.range(probs.shape[1]),
                                 self.tgt_vocab.token2idx[self.tgt_vocab.PAD]),
                        dtype=tf.float32) * -1e9, 0),
            tf.reshape(tf.constant([batch_size * beam_size, 1]),
                       [-1])) * finalized_hypos_mask_reverse
        probs += tf.expand_dims(scores[:, t - 1], 1)
      else:
        probs = tf.gather_nd(probs, bbsz_offset)
        probs += tf.cast(tf.equal(tf.range(probs.shape[1]),
                                  self.tgt_vocab.token2idx[self.tgt_vocab.PAD]),
                         dtype=tf.float32) * -1e9
      cand_scores, cand_indices = tf.math.top_k(tf.reshape(
          probs, [batch_size, -1]),
                                                k=beam_size)
      cand_beams = cand_indices // (self.tgt_vocab_size + self.src_seq_len)
      cand_indices = tf.math.floormod(cand_indices,
                                      (self.tgt_vocab_size + self.src_seq_len))
      cand_bbsz_idx = tf.reshape(cand_beams + bbsz_offset,
                                 [-1, 1])  # bsz x (2 x beam)
      active_tokens = tf.gather_nd(tgt_inputs,
                                   indices=cand_bbsz_idx)  # (bsz x beam) x T
      step_tokens = tf.reshape(cand_indices, [-1, 1])

      tgt_inputs = tf.concat([active_tokens, step_tokens], axis=1)
      step_scores = tf.reshape(cand_scores, [-1, 1])

      active_scores = tf.gather_nd(scores, indices=cand_bbsz_idx)
      scores = tf.concat([active_scores, step_scores], axis=1)

      reorder_state = cand_bbsz_idx

    return tf.reshape(tgt_inputs, [batch_size, beam_size, -1]), tf.reshape(
        scores, [batch_size, beam_size, -1])
