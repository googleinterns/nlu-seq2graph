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

from absl import app
from absl import flags

import os
import json
import sys
import time

import tensorflow as tf

from vocabulary import Vocabulary
from dataset import build_dataset
from transformer import Transformer
from graph_utils import Node
from graph_utils import SignatureGraph
from graph_utils import is_valid_sequence
from graph_utils import is_valid_tree
from graph_utils import tok_to_tree
from graph_utils import calc_signature
from graph_utils import graph_to_tfex

flags.DEFINE_string("data_spec", None, "Path to training data spec.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("beam_size", 1, "beam size.")
flags.DEFINE_integer("graph_size", 1, "beam size.")
flags.DEFINE_string(
    "predict", None,
    "Init model from save_model_path and run prediction on the data set,")
flags.DEFINE_string("save_model_path", None, "Save model path.")
flags.DEFINE_bool("eager_run", False, "Run in eager mode for debugging.")
flags.DEFINE_string("output_path", None, "Output TF examples.")
flags.DEFINE_string("output_derivs", None, "Output derivations.")
FLAGS = flags.FLAGS


def get_top_beam_graphs(model_type,
                        eval_set,
                        src_vocab,
                        tgt_vocab,
                        save_model_path,
                        output_record,
                        graph_size=1):

  hparams = json.load(open(os.path.join(save_model_path, "hparams.json")))
  hparams["predict_edge"] = False

  model = model_type(src_vocab, tgt_vocab, hparams)
  model.load_weights(os.path.join(save_model_path, "model_weights"))

  dev_start = time.time()
  total_tree_size = 0
  total_merge_size = 0

  total_cycle_count = 0

  results = []
  max_src_length = 0
  max_tgt_length = 0
  self_loop_refs = []

  all_derivs = []
  for batch, examples in enumerate(eval_set):
    print("batch", batch)
    batch_tree_size = 0
    batch_merge_size = 0
    predictions, _ = model(examples,
                           beam_size=FLAGS.beam_size,
                           is_train=False,
                           return_all=True)
    predictions = predictions.numpy().tolist()
    for exid in range(examples["src_token_ids"].shape[0]):
      src_token_ids = examples["src_token_ids"][exid]
      src_tokens = [src_vocab.idx2token[idx] for idx in src_token_ids]
      src_length = sum(tok != src_vocab.token2idx[src_vocab.PAD]
                       for tok in src_token_ids.numpy().tolist())

      tgt_token_ids = examples["tgt_token_ids"][exid].numpy().tolist()
      tgt_length = sum(
          tok != tgt_vocab.token2idx[tgt_vocab.PAD] for tok in tgt_token_ids)
      tgt_tokens = [
          tgt_vocab.idx2token[idx]
          if idx < len(tgt_vocab) else str(idx - len(tgt_vocab))
          for idx in tgt_token_ids[:tgt_length]
      ]
      print("ref", tgt_tokens)

      graphs = []
      flatten_derivs = []

      if tgt_tokens[0] == tgt_vocab.BOS:
        tgt_tokens = tgt_tokens[1:]
      if tgt_tokens[-1] == tgt_vocab.EOS:
        tgt_tokens = tgt_tokens[:-1]

      root = Node(tgt_vocab.BOS)
      tgt_nodes = tok_to_tree(tgt_tokens, root)
      calc_signature(root)
      graphs.append(tgt_nodes)
      sig_graph = SignatureGraph(root.signature)
      if not sig_graph.try_adding_tree(tgt_nodes):
        print("!!! invalid ref tree")
        self_loop_refs.append(tgt_nodes)

      flatten_derivs.append(" ".join(tgt_tokens))

      for sent in predictions[exid]:
        pred_tokens = sent
        if tgt_vocab.token2idx[tgt_vocab.EOS] in pred_tokens:
          dev_length = pred_tokens.index(tgt_vocab.token2idx[tgt_vocab.EOS])
        else:
          dev_length = len(pred_tokens)
        pred_tokens = pred_tokens[:dev_length]

        pred_tokens_words = [
            tgt_vocab.idx2token[idx]
            if idx < len(tgt_vocab) else src_tokens[idx - len(tgt_vocab)]
            for idx in pred_tokens
        ]
        if is_valid_sequence(pred_tokens_words, src_tokens, pred_tokens,
                             tgt_vocab):
          for j in range(len(pred_tokens)):
            if pred_tokens[j] >= len(tgt_vocab):
              pred_tokens_words[j] = str(pred_tokens[j] - len(tgt_vocab))
          if pred_tokens_words[0] == tgt_vocab.BOS:
            pred_tokens_words = pred_tokens_words[1:]
          if pred_tokens_words[-1] == tgt_vocab.EOS:
            pred_tokens_words = pred_tokens_words[:-1]
          root = Node(tgt_vocab.BOS)
          pred_tokens_nodes = tok_to_tree(pred_tokens_words, root)

          if not is_valid_tree(pred_tokens_nodes):
            print("invalid tree", pred_tokens_words)
            continue

          print("Attempt to merge", pred_tokens_words)
          calc_signature(root)
          assert root.signature == sig_graph.root_signature
          if sig_graph.try_adding_tree(pred_tokens_nodes):
            print("Added")
            graphs.append(pred_tokens_nodes)
            flatten_derivs.append(" ".join(pred_tokens_words))
          else:
            print("Failed")

        if len(graphs) == graph_size + 1:
          break
      for g in graphs:
        print("pre", " ".join([v.word for v in g]))

      sorted_nodes = sig_graph.topo_sort()
      print("topo sort", sorted_nodes)

      for g in graphs:
        batch_tree_size += (len(g) + 1)
        total_tree_size += (len(g) + 1)

      batch_merge_size += len(sig_graph.nodes)
      total_merge_size += len(sig_graph.nodes)

      graph_tgt_toks, graph_tgt_edges = graph_to_tfex(sorted_nodes, sig_graph,
                                                      tgt_vocab)
      graph_tgt_tok_ids = []

      for tok in graph_tgt_toks:
        if tok in tgt_vocab:
          graph_tgt_tok_ids.append(tgt_vocab.token2idx[tok])
        elif tok.isnumeric():
          graph_tgt_tok_ids.append(len(tgt_vocab) + int(tok))
        else:
          assert False
      print(graph_tgt_toks)
      print(graph_tgt_tok_ids)
      print(graph_tgt_edges)
      print("tgt len", len(graph_tgt_tok_ids), "max tgt len", max_tgt_length)
      assert (len(graph_tgt_tok_ids) == len(graph_tgt_toks) ==
              len(graph_tgt_edges))
      if len(graph_tgt_tok_ids) > max_tgt_length:
        max_tgt_length = len(graph_tgt_tok_ids)
        print("new_max_tgt_length", max_tgt_length)
        print("tgt_toks", graph_tgt_toks)
        print("src_toks", src_tokens)
        print("src_tok_length", src_length)
      if src_length > max_src_length:
        max_src_length = src_length
        print("new_max_src_length", max_src_length)
        print("tgt_toks", graph_tgt_toks)
        print("src_toks", src_tokens)
        print("tgt_tok_length", len(graph_tgt_tok_ids))

      # Sanity check
      graph_tgt_tokens_copied = [
          i for i in graph_tgt_tok_ids if i >= len(tgt_vocab)
      ]
      src_tokens_no_pad = [i for i in src_tokens if i != src_vocab.PAD]
      if len(graph_tgt_tokens_copied) != len(src_tokens_no_pad):
        print("Mismatch tgt copied length")
        print(graph_tgt_tokens_copied)
        print(src_tokens_no_pad)
        sys.exit(-1)
      passed = True
      for i in range(1, len(graph_tgt_tokens_copied)):
        if graph_tgt_tokens_copied[i] != graph_tgt_tokens_copied[i - 1] + 1:
          passed = False
          break
      if not passed:
        print("Incorrect tgt order", graph_tgt_tokens_copied)
        sys.exit(-1)
      # reconstruct the ref tree
      ref_root = Node(tgt_vocab.BOS)
      ref_nodes = tok_to_tree(tgt_tokens, ref_root)
      passed = sig_graph.contains_tree(ref_nodes)
      if not passed:
        print("Reference tree not included in graph")
        print("Refrence tree", ref_nodes)
        sys.exit(-1)

      features = {}

      features["src_tokens"] = tf.train.Feature(bytes_list=tf.train.BytesList(
          value=(tok.encode("utf-8") for tok in src_tokens[:src_length])))
      features["src_token_ids"] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=src_token_ids[:src_length]))
      features["tgt_tokens"] = tf.train.Feature(bytes_list=tf.train.BytesList(
          value=(tok.encode("utf-8") for tok in graph_tgt_toks)))
      features["tgt_token_ids"] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=graph_tgt_tok_ids))
      features["tgt_edges"] = tf.train.Feature(int64_list=tf.train.Int64List(
          value=graph_tgt_edges.reshape((-1))))
      features["tgt_orig_token_ids"] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=examples["tgt_token_ids"][exid]))
      all_derivs.append(flatten_derivs)

      example = tf.train.Example(features=tf.train.Features(feature=features))
      results.append(example)

    print("batch total size: {}, merge size: {}".format(batch_tree_size,
                                                        batch_merge_size))

    if batch % 10 == 0:
      print("Batch {}, Time {}s".format(batch, time.time() - dev_start))
  print("total size: {}, merge size: {}".format(total_tree_size,
                                                total_merge_size))
  print("total # of examples with cycles", total_cycle_count)
  if FLAGS.output_derivs:
    json.dump(all_derivs, open(FLAGS.output_derivs, "w"))
  with tf.io.TFRecordWriter(output_record) as record_writer:
    count = 0
    max_src_len = 0
    max_tgt_len = 0
    for example in results:
      max_src_len = max(
          max_src_len,
          len(example.features.feature["src_token_ids"].int64_list.value))
      max_tgt_len = max(
          max_tgt_len,
          len(example.features.feature["tgt_token_ids"].int64_list.value))
      record_writer.write(example.SerializeToString())
      count += 1
    print(f"{count} examples added to {output_record}.",
          f"Max src length {max_src_len}, max tgt length {max_tgt_length}.")


def main(argv):
  if FLAGS.eager_run:
    tf.config.experimental_run_functions_eagerly(True)

  model_type = Transformer

  data_spec = json.load(open(FLAGS.data_spec))

  src_vocab = Vocabulary.load(data_spec["source_vocab"])
  tgt_vocab = Vocabulary.load(data_spec["target_vocab"])
  eval_set = build_dataset(os.path.join(FLAGS.predict),
                           max_num_src_tokens=data_spec["max_num_src_tokens"],
                           max_num_tgt_tokens=data_spec["max_num_tgt_tokens"],
                           src_vocab=src_vocab,
                           tgt_vocab=tgt_vocab,
                           is_train=False)
  eval_set = eval_set.batch(FLAGS.batch_size, drop_remainder=False)
  get_top_beam_graphs(model_type,
                      eval_set,
                      src_vocab,
                      tgt_vocab,
                      FLAGS.save_model_path,
                      FLAGS.output_path,
                      graph_size=FLAGS.graph_size)


if __name__ == "__main__":
  app.run(main)
