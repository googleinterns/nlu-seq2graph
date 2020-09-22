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
"""Generate Task Oriented Dialog to TF examples."""

from absl import app
from absl import flags
import json
import os

import tensorflow as tf

from vocabulary import Vocabulary
from vocabulary import VocabularyBuilder
from utils import fetch_examples
from graph_utils import Node
from graph_utils import tok_to_tree
from graph_utils import tree_to_data

flags.DEFINE_string("data_spec", None, "Path to training data spec.")
flags.DEFINE_string("tod_folder", None, "Path to TOD dataset.")
flags.DEFINE_string("output_folder", None, "Path to output TF examples.")
flags.DEFINE_bool("predict_edge", False, "use this flag for seq2graph")
flags.DEFINE_bool("multiple", False, "use this flag for seq2graph")

FLAGS = flags.FLAGS


def process_queries(example_file, src_vocab, tgt_vocab):
  for original_query, src_toks_orig, tgt_toks_orig in fetch_examples(
      example_file):
    src_toks = [
        tok if tok in src_vocab else src_vocab.OOV for tok in src_toks_orig
    ]
    src_tok_ids = src_vocab.convert(src_toks, add_bos_eos=False)
    tgt_toks = [tgt_vocab.BOS]
    tgt_tok_ids = [tgt_vocab.token2idx[tgt_vocab.BOS]]
    src_index = 0
    for tok in tgt_toks_orig:
      if tok in tgt_vocab:
        tgt_toks.append(tok)
        tgt_tok_ids.append(tgt_vocab.token2idx[tok])
      elif tok in src_toks_orig:
        # For copied src tokens, the id will be the index of the token in the
        # src query + an offset of the tgt vocab size
        assert tok == src_toks_orig[src_index]
        tgt_toks.append(tok)
        tgt_tok_ids.append(len(tgt_vocab) + src_index)
        src_index += 1
      else:
        print(original_query, tgt_toks_orig, tok)
        assert False
    assert src_index == len(src_toks_orig)
    tgt_toks.append(tgt_vocab.EOS)
    tgt_tok_ids.append(tgt_vocab.token2idx[tgt_vocab.EOS])

    features = {}
    features["src_tokens_original"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tok.encode("utf-8") for tok in src_toks_orig]))
    features["src_tokens"] = tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tok.encode("utf-8") for tok in src_toks]))
    features["src_token_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=src_tok_ids))
    features["tgt_tokens_original"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tok.encode("utf-8") for tok in tgt_toks_orig]))
    features["tgt_tokens"] = tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tok.encode("utf-8") for tok in tgt_toks]))
    features["tgt_token_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=tgt_tok_ids))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    yield example


def process_query_graph(example_file, src_vocab, tgt_vocab, multiple=False):
  for original_query, src_toks_orig, tgt_toks_orig in fetch_examples(
      example_file):
    src_toks = [
        tok if tok in src_vocab else src_vocab.OOV for tok in src_toks_orig
    ]
    src_tok_ids = src_vocab.convert(src_toks, add_bos_eos=False)
    root = Node(tgt_vocab.BOS, None)
    tgt_nodes = tok_to_tree(tgt_toks_orig, root)
    eos = Node(tgt_vocab.EOS, root, index=len(tgt_nodes))
    tgt_nodes.append(eos)
    tgt_nodes[0].child.append(eos)
    tgt_toks, tgt_edges = tree_to_data(tgt_nodes, multiple=multiple)
    print(tgt_toks_orig)
    print(tgt_toks)
    print(tgt_edges)
    tgt_tok_ids = []
    src_index = 0
    for tok in tgt_toks:
      if tok in tgt_vocab:
        tgt_tok_ids.append(tgt_vocab.token2idx[tok])
      elif tok in src_toks_orig:
        assert tok == src_toks_orig[src_index]
        tgt_tok_ids.append(len(tgt_vocab) + src_index)
        src_index += 1
      else:
        print(original_query, tgt_toks_orig, tok)
        assert False
    assert src_index == len(src_toks_orig)
    assert len(tgt_tok_ids) == len(tgt_toks) == len(tgt_edges)
    features = {}
    features["src_tokens_original"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tok.encode("utf-8") for tok in src_toks_orig]))
    features["src_tokens"] = tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tok.encode("utf-8") for tok in src_toks]))
    features["src_token_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=src_tok_ids))
    features["tgt_tokens_original"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[tok.encode("utf-8") for tok in tgt_toks_orig]))
    features["tgt_tokens"] = tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tok.encode("utf-8") for tok in tgt_toks]))
    features["tgt_token_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(
        value=tgt_tok_ids))
    if multiple:
      features["tgt_edges"] = tf.train.Feature(int64_list=tf.train.Int64List(
          value=tgt_edges.reshape(-1)))
    else:
      features["tgt_edges"] = tf.train.Feature(int64_list=tf.train.Int64List(
          value=tgt_edges))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    yield example


def write_record(example_file, src_vocab, tgt_vocab, output_record):
  with tf.io.TFRecordWriter(output_record) as record_writer:
    count = 0
    max_src_len = 0
    max_tgt_len = 0
    for example in process_queries(example_file, src_vocab, tgt_vocab):
      max_src_len = max(
          max_src_len,
          len(example.features.feature["src_token_ids"].int64_list.value))
      max_tgt_len = max(
          max_tgt_len,
          len(example.features.feature["tgt_token_ids"].int64_list.value))
      record_writer.write(example.SerializeToString())
      count += 1
    print(f"{count} examples added to {output_record}.",
          f"Max src length {max_src_len}, max tgt length {max_tgt_len}.")


def write_record_graph(example_file, src_vocab, tgt_vocab, output_record):
  with tf.io.TFRecordWriter(output_record) as record_writer:
    count = 0
    max_src_len = 0
    max_tgt_len = 0
    for example in process_query_graph(example_file,
                                       src_vocab,
                                       tgt_vocab,
                                       multiple=FLAGS.multiple):
      max_src_len = max(
          max_src_len,
          len(example.features.feature["src_token_ids"].int64_list.value))
      max_tgt_len = max(
          max_tgt_len,
          len(example.features.feature["tgt_token_ids"].int64_list.value))
      record_writer.write(example.SerializeToString())
      count += 1
    print(f"{count} examples added to {output_record}.",
          f"Max src length {max_src_len}, max tgt length {max_tgt_len}.")


def main(argv):
  del argv  # Unused.
  data_spec = json.load(open(FLAGS.data_spec))
  src_vocab = Vocabulary.load(data_spec["source_vocab"])
  tgt_vocab = Vocabulary.load(data_spec["target_vocab"])
  print(f"{len(src_vocab)} srouce tokens and {len(tgt_vocab)} target tokens.")

  if FLAGS.predict_edge:
    write_record_graph(
        os.path.join(FLAGS.tod_folder, "train.tsv"), src_vocab, tgt_vocab,
        os.path.join(FLAGS.output_folder, "train_graph_multiple_sort.record"))
    write_record_graph(
        os.path.join(FLAGS.tod_folder, "eval.tsv"), src_vocab, tgt_vocab,
        os.path.join(FLAGS.output_folder, "dev_graph_multiple_sort.record"))
    write_record_graph(
        os.path.join(FLAGS.tod_folder, "test.tsv"), src_vocab, tgt_vocab,
        os.path.join(FLAGS.output_folder, "test_graph_multiple_sort.record"))
  else:
    write_record(os.path.join(FLAGS.tod_folder, "train.tsv"), src_vocab,
                 tgt_vocab, os.path.join(FLAGS.output_folder, "train.record"))
    write_record(os.path.join(FLAGS.tod_folder, "eval.tsv"), src_vocab,
                 tgt_vocab, os.path.join(FLAGS.output_folder, "dev.record"))
    write_record(os.path.join(FLAGS.tod_folder, "test.tsv"), src_vocab,
                 tgt_vocab, os.path.join(FLAGS.output_folder, "test.record"))


if __name__ == "__main__":
  app.run(main)
