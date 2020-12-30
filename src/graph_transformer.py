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
"""Transformer SeqGraph model."""
import math
import numpy as np

import tensorflow as tf

from transformer import get_angles
from transformer import positional_encoding
from transformer import scaled_dot_product_attention
from transformer import create_padding_mask
from transformer import create_look_ahead_mask
from transformer import create_masks
from transformer import MultiHeadAttention
from transformer import point_wise_feed_forward_network
from transformer import Encoder
from transformer import DecoderLayer


class GraphDecoder(tf.keras.layers.Layer):
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
               biaffine=True):
    super(GraphDecoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.tgt_vocab = tgt_vocab
    self.tgt_vocab_size = len(tgt_vocab)
    self.src_seq_len = src_seq_len
    self.biaffine = biaffine

    self.embedding = tf.keras.layers.Embedding(self.tgt_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [
        DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
    ]

    self.gen_scores_layer = tf.keras.layers.Dense(self.tgt_vocab_size)
    self.copy_scores_proj_layer = tf.keras.layers.Dense(self.d_model,
                                                        activation="tanh")

    self.dropout = tf.keras.layers.Dropout(rate)
    self.input_edge_w = tf.keras.layers.Dense(self.d_model, activation="tanh")
    self.input_edge_v = tf.keras.layers.Dense(1)
    self.merge_embedding = tf.keras.layers.Dense(self.d_model,
                                                 activation="tanh")
    self.edge_q = tf.keras.layers.Dense(self.d_model, activation="tanh")
    self.edge_k = tf.keras.layers.Dense(self.d_model, activation="tanh")
    if self.biaffine:
      w_initializer = tf.keras.initializers.Orthogonal()
      self.biaffine_w_arc = tf.Variable(initial_value=w_initializer(
          shape=(1, self.d_model, self.d_model + 1), dtype=tf.float32),
                                        trainable=True)

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
    # x.shape: [batch_size, seq_len]
    # tgt_edges.shape: [batch_size, seq_len, tgt_seq_len]

    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    attention_weights = {}

    if mem is not None:
      x = tf.expand_dims(x[:, -1], -1)

    # (batch_size, tgt_seq_len, d_model)
    x = self._get_input_embedding(x, enc_output)
    x_ = None
    if mem is not None:
      if -1 in mem:
        mem[-1] = tf.concat([mem[-1], x], axis=1)
        x_ = mem[-1][:, :-1]
      else:
        mem[-1] = x
        x_ = mem[-1]  # ?

    if tgt_edges is not None:
      tgt_edges = tf.cast(tgt_edges, dtype=tf.float32)
      if mem is not None:
        if seq_len == 1:
          edge_x = tf.zeros([batch_size, 1, self.d_model])
        else:
          # [batch, 1, tgt_seq_len]
          tgt_edges = tf.expand_dims(tgt_edges[:, -1,  :], 1)
          # [batch, 1, seq_len-1]
          att_logit = tf.squeeze(
              self.input_edge_v(self.input_edge_w(tf.expand_dims(x_, 1))), -1)
          # mask out non-exist edges
          att_logit += (1 - tgt_edges[:, :, :(seq_len - 1)]) * -1e9
          att_score = tf.nn.softmax(att_logit,
                                    axis=-1) * tgt_edges[:, :, :(seq_len - 1)]
          edge_x = tf.matmul(att_score, x_)
      else:
        att_logit = tf.squeeze(
            self.input_edge_v(
                self.input_edge_w(
                    tf.repeat(tf.expand_dims(x, 1), seq_len, axis=1))), -1)
        att_logit += (1 - tgt_edges) * -1e9
        att_score = tf.nn.softmax(att_logit, axis=-1) * tgt_edges
        edge_x = tf.matmul(att_score, x)

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
      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

    # TODO: should we consider attentions here?
    # (batch_size, tgt_seq_len, tgt_vocab_size)
    gen_scores = self._gen_scores(x)
    # (batch_sz, tgt_seq_len, src_seq_len)
    copy_scores = self._copy_scores(x, enc_output)

    scores = tf.concat([gen_scores, copy_scores], axis=2)

    q = self.edge_q(x)
    if mem is not None:
      k = self.edge_k(mem[self.num_layers])
    else:
      k = self.edge_k(x)
    if self.biaffine:
      ones = tf.ones([batch_size, tf.shape(k)[1], 1], dtype=k.dtype)
      # (batch_size, t, dim+1)
      extended_k = tf.concat([k, ones], axis=-1)
      # (batch_size, 1, dim+1)
      edge_scores = tf.matmul(q, self.biaffine_w_arc)
      # (batch_size, 1, t)
      edge_scores = tf.matmul(edge_scores, extended_k, transpose_b=True)
    else:
      edge_scores = tf.matmul(q, k, transpose_b=True)
    if mem is None:
      edge_scores += (tf.squeeze(look_ahead_mask, 1) * -1e9)
    return scores, attention_weights, edge_scores


class GraphTransformer(tf.keras.Model):
  default_hparams = {
      "num_layers": 3,
      "d_model": 128,
      "num_heads": 8,
      "dff": 512,
      "num_src_tokens": None,
      "num_tgt_tokens": None,
      "dropout": 0.2,
      "biaffine": True
  }

  def __init__(self, src_vocab, tgt_vocab, hparams=None):
    super(GraphTransformer, self).__init__()
    self.hparams = dict(GraphTransformer.default_hparams)
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
    self.biaffine = self.hparams["biaffine"]

    self.encoder = Encoder(num_layers=self.hparams["num_layers"],
                           d_model=self.hparams["d_model"],
                           num_heads=self.hparams["num_heads"],
                           dff=self.hparams["dff"],
                           source_vocab_size=self.src_vocab_size,
                           maximum_position_encoding=self.src_seq_len,
                           rate=self.hparams["dropout"])

    self.decoder = GraphDecoder(num_layers=self.hparams["num_layers"],
                                d_model=self.hparams["d_model"],
                                num_heads=self.hparams["num_heads"],
                                dff=self.hparams["dff"],
                                tgt_vocab=self.tgt_vocab,
                                src_seq_len=self.src_seq_len,
                                maximum_position_encoding=self.tgt_seq_len,
                                rate=self.hparams["dropout"],
                                biaffine=self.biaffine)

  def call(self, examples, is_train=True):
    if is_train:
      return self.train(examples, is_train)
    else:
      return self.greedy_decode(examples, is_train)

  def train(self, examples, is_train=True):
    src_token_ids = examples["src_token_ids"]
    tgt_token_ids = examples["tgt_token_ids"]
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
    dec_output, _, edge_scores = self.decoder(tgt_token_ids,
                                              enc_output,
                                              is_train,
                                              combined_mask,
                                              dec_padding_mask,
                                              tgt_edges=tgt_edges)
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
    return dec_output, edge_scores

  def greedy_decode(self, examples, is_train=False, tgt_seq_len=None):
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
    tgt_edges = tf.zeros([batch_size, 1, tgt_seq_len], dtype=tf.int64) - 1

    start_token_onehot = tf.one_hot(start_token,
                                    depth=(self.tgt_vocab_size +
                                           self.src_seq_len))
    start_token_logits = start_token_onehot + (start_token_onehot - 1) * 1e9
    output = [start_token_logits]
    edge_output = []

    mem = {}

    for t in range(1, tgt_seq_len):
      look_ahead_mask = create_look_ahead_mask(t)[tf.newaxis, tf.newaxis, :, :]
      # dec_output.shape == (batch_sz, t, tgt_vocab_size+src_seq_len)
      dec_output, _, edge_scores = self.decoder(tgt_inputs,
                                                enc_output,
                                                is_train,
                                                look_ahead_mask,
                                                dec_padding_mask,
                                                mem=mem,
                                                tgt_edges=tgt_edges)

      # (batch_sz, tgt_vocab_size+src_seq_len)
      last_step_output = dec_output[:, -1, :]
      last_step_output_idx = tf.expand_dims(tf.argmax(last_step_output, axis=1),
                                            axis=-1)
      tgt_inputs = tf.concat([tgt_inputs, last_step_output_idx], axis=-1)

      last_step_score = tf.expand_dims(edge_scores[:, -1, :], 1)
      last_step_score_idx = tf.cast(last_step_score > 0, tf.int64)
      pad = tf.zeros([batch_size, 1, tgt_seq_len - t], dtype=tf.int64)
      last_step_score_idx = tf.concat([last_step_score_idx, pad], axis=-1)
      tgt_edges = tf.concat([tgt_edges, last_step_score_idx], axis=1)

      # (batch_sz, t+1)
      output.append(dec_output)

      edge_output.append(
          tf.concat(
              [edge_scores,
               tf.fill([batch_size, 1, tgt_seq_len - t], -1e9)],
              axis=2))

    dec_output = tf.concat(output, axis=1)
    edge_output.append(tf.fill([batch_size, 1, tgt_seq_len], -1e9))
    edge_output = tf.concat(edge_output, axis=1)
    return dec_output, edge_output
