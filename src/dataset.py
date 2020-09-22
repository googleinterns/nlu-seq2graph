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
"""dataset utils."""
from absl import app
import functools

import tensorflow as tf

TF_EXAMPLE_SPEC = {
    "src_token_ids":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "tgt_token_ids":
        tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
}

_NUM_THREADS = 16


def _truncate_and_pad_dataset(dataset, max_num_src_tokens, max_num_tgt_tokens,
                              src_pad_id, tgt_pad_id, predict_edge=False, multiple=False):
  """Truncate dataset to max_len for static shapes."""

  def truncate_pad_to_length(tensor, length, pad_id):
    tensor = tensor[:length]
    tensor = tf.reshape(tensor, [1, -1])
    tensor = tf.pad(
        tensor, [[0, 0], [0, length - tf.shape(tensor)[1]]],
        mode="CONSTANT",
        constant_values=pad_id)
    return tf.reshape(tensor, [-1])

  if predict_edge:
    if multiple:
      tensor = dataset["tgt_edges"]
      length = len(dataset["tgt_token_ids"])
      tensor = tf.reshape(tensor, [length, -1])
      pad1 = tf.zeros([max_num_tgt_tokens-length, length], dtype=tf.int64)
      pad2 = tf.zeros([max_num_tgt_tokens, max_num_tgt_tokens-length], dtype=tf.int64)
      dataset["tgt_edges"] = tf.concat([tf.concat([tensor, pad1], axis=0), pad2], axis=1)
      #tf.print(dataset["tgt_edges"])
    else:
      dataset["tgt_edges"] = truncate_pad_to_length(dataset["tgt_edges"],
                                                    max_num_tgt_tokens,
                                                    -1)

  dataset["src_token_ids"] = truncate_pad_to_length(dataset["src_token_ids"],
                                                    max_num_src_tokens,
                                                    src_pad_id)
  dataset["tgt_token_ids"] = truncate_pad_to_length(dataset["tgt_token_ids"],
                                                    max_num_tgt_tokens,
                                                    tgt_pad_id)

  return dataset


def _remove_invalid_copy_src_ids(dataset, max_num_src_tokens, tgt_vocab):
  """Replace src indices >= max_num_src_tokens with tgt OOV."""

  def _item_assign(tensor, select, value):
    select = tf.cast(select, dtype=tensor.dtype)
    return tensor * (1 - select) + value * tf.ones_like(tensor) * select

  is_from_src_vocab = dataset["tgt_token_ids"] >= len(tgt_vocab)
  tgt_oov_id = tgt_vocab.token2idx[tgt_vocab.OOV]
  invalid_src_ids = tf.logical_and(
      is_from_src_vocab,
      (dataset["tgt_token_ids"] - len(tgt_vocab) >= max_num_src_tokens))
  dataset["tgt_token_ids"] = _item_assign(dataset["tgt_token_ids"],
                                          invalid_src_ids, tgt_oov_id)
  return dataset


def build_dataset(file_pattern, max_num_src_tokens, max_num_tgt_tokens,
                  src_vocab, tgt_vocab, is_train, predict_edge=False, multiple=False):
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_train)
  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  if predict_edge:
    TF_EXAMPLE_SPEC["tgt_edges"] = tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
  parse_mapper = functools.partial(
      tf.io.parse_single_example, features=TF_EXAMPLE_SPEC)
  # TF requires this statement to be here
  anon_fn = lambda x: parse_mapper(x)
  dataset = dataset.map(anon_fn, num_parallel_calls=_NUM_THREADS)
  src_pad_id = src_vocab.token2idx[src_vocab.PAD]
  tgt_pad_id = tgt_vocab.token2idx[tgt_vocab.PAD]
  tgt_oov_id = tgt_vocab.token2idx[tgt_vocab.OOV]
  truncate = functools.partial(
      _truncate_and_pad_dataset,
      max_num_src_tokens=max_num_src_tokens,
      max_num_tgt_tokens=max_num_tgt_tokens,
      src_pad_id=src_pad_id,
      tgt_pad_id=tgt_pad_id,
      predict_edge=predict_edge,
      multiple=multiple)
  dataset = dataset.map(truncate, num_parallel_calls=_NUM_THREADS)
  remove_invalid = functools.partial(
      _remove_invalid_copy_src_ids,
      max_num_src_tokens=max_num_src_tokens,
      tgt_vocab=tgt_vocab)
  dataset = dataset.map(remove_invalid, num_parallel_calls=_NUM_THREADS)

  return dataset
