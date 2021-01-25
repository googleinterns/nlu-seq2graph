"""Copyright 2020 Google LLC Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""Trainer.

Usage:
Training:
trainer --data_folder ~/workspace/seq2graph/spider \
  --epochs 500 --save_path ~/workspace/seq2graph/seq2seq_savedmodel

Predicting:
trainer --save_model_path ~/workspace/seq2graph/seq2seq_savedmodel \
  --predict ~/workspace/seq2graph/spider/train.record \
  --predict_output ~/tmp/seq2seq_train.txt
"""

from absl import app
from absl import flags
from absl import logging

import os
import json
import sys
import shutil
import time

import tensorflow as tf
import numpy as np

from vocabulary import Vocabulary
from dataset import build_dataset
from graph_transformer import GraphTransformer
from graph_utils import contains_tree
from graph_utils import reconstruct_tree
from graph_utils import precompute_children_combinations
from graph_utils import retrieve_trees
from training_utils import NoamSchedule
from training_utils import EdgeLoss
from training_utils import SequenceLoss
from fscore import FScore

flags.DEFINE_string("data_spec", None, "Path to training data spec.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("model_dim", 128, "Model dim.")
flags.DEFINE_integer("epochs", 10, "Num of epochs.")
flags.DEFINE_float("dropout", 0.2, "Dropout rate.")
flags.DEFINE_bool("biaffine", True, "Use Biaffine in edge prediction.")
flags.DEFINE_string("save_model_path", None, "Save model path.")
flags.DEFINE_string(
    "predict", None,
    "Init model from save_model_path and run prediction on the data set,")
flags.DEFINE_string("predict_output", None, "Prediction output.")
flags.DEFINE_bool("eager_run", False, "Run in eager mode for debugging.")
flags.DEFINE_string("ref_derivs", None, "Reference dev derivations.")

FLAGS = flags.FLAGS


@tf.function
def process_one_batch(model,
                      sequence_loss_fn,
                      edge_loss_fn,
                      examples,
                      tgt_vocab,
                      optimizer,
                      token_accuracy,
                      exact_accuracy,
                      is_train=True,
                      edge_accuracy=None):
  with tf.GradientTape() as tape:
    # Shape: [batch_sz, max_num_tgt_tokens, tgt_vocab_size + max_num_src_tokens]
    tgt_token_ids = examples["tgt_token_ids"]
    tgt_edges = examples["tgt_edges"]
    prediction_logits, edge_logits = model(examples, is_train=is_train)
    loss = sequence_loss_fn(prediction_logits, tgt_token_ids)
    m = tf.math.not_equal(tgt_token_ids, tgt_vocab.token2idx[tgt_vocab.PAD])
    loss += edge_loss_fn(edge_logits[:, :-1, :],
                         tgt_edges[:, 1:, :],
                         edge_mask=m)
    prediction_edge = tf.cast(edge_logits > 0, dtype=tf.int64)

    predictions = tf.argmax(prediction_logits, axis=-1)

    if is_train:
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if is_train:

      # Shape: [batch_sz, max_num_tgt_tokens]
      oov_mask = tf.not_equal(tgt_token_ids, tgt_vocab.token2idx[tgt_vocab.PAD])

      token_accuracy.update_state(tgt_token_ids,
                                  predictions,
                                  sample_weight=tf.cast(oov_mask,
                                                        dtype=tf.float32))

      edge_accuracy.update_state(tgt_edges[:, 1:, :],
                                 prediction_edge[:, :-1, :],
                                 sample_weight=tf.cast(oov_mask[:, 1:],
                                                       dtype=tf.float32))
      false_positive = tf.reduce_sum(
          tf.cast((prediction_edge[:, :-1, :] - tgt_edges[:, 1:, :]) == 1,
                  dtype=tf.int64) *
          tf.cast(oov_mask[:, 1:, tf.newaxis], dtype=tf.int64))
      false_negative = tf.reduce_sum(
          tf.cast((tgt_edges[:, 1:, :] - prediction_edge[:, :-1, :]) == 1,
                  dtype=tf.int64) *
          tf.cast(oov_mask[:, 1:, tf.newaxis], dtype=tf.int64))
      true_positive = tf.reduce_sum(
          tf.cast((tgt_edges[:, 1:, :] + prediction_edge[:, :-1, :]) == 2,
                  dtype=tf.int64) *
          tf.cast(oov_mask[:, 1:, tf.newaxis], dtype=tf.int64))
      precision = true_positive / (true_positive + false_positive)
      recall = true_positive / (true_positive + false_negative)
      f1 = 2 * precision * recall / (precision + recall)

      exact_match = tf.cast(
          tf.reduce_all(tf.concat([
              tf.logical_or(tf.equal(tgt_token_ids, predictions),
                            tf.logical_not(oov_mask)),
              tf.logical_or(
                  tf.reduce_all(tf.equal(tgt_edges[:, 1:, :],
                                         prediction_edge[:, :-1, :]),
                                axis=2), tf.logical_not(oov_mask[:, 1:]))
          ],
                                  axis=1),
                        axis=1), tf.float32)

      exact_accuracy.update_state(tf.ones_like(exact_match), exact_match)
      return loss, predictions, prediction_edge, precision, recall, f1
    else:
      return loss, predictions, prediction_edge, None, None, None


def get_ref_tree_strings(flatten_tgt_trees, tgt_vocab):
  ref_trees = [[]]
  for v in flatten_tgt_trees:
    if v == tgt_vocab.token2idx[tgt_vocab.PAD]:
      ref_trees.append([])
    else:
      ref_trees[-1].append(
          tgt_vocab.idx2token[v] if v < len(tgt_vocab) else str(v -
                                                                len(tgt_vocab)))
  return [" ".join(t) for t in ref_trees if t]


def train(model_type,
          epochs,
          train_set,
          dev_set,
          src_vocab,
          tgt_vocab,
          hparams,
          save_model_path,
          log_error=False):
  optimizer = tf.keras.optimizers.Adam(
      learning_rate=NoamSchedule(FLAGS.model_dim))
  model = model_type(src_vocab, tgt_vocab, hparams)
  sequence_loss_fn = SequenceLoss(tgt_vocab)
  edge_loss_fn = EdgeLoss()

  error_log = []

  if save_model_path:
    try:
      shutil.rmtree(save_model_path)
    except:
      pass
    os.mkdir(save_model_path)

  best_token_accuracy_train = 0
  best_token_accuracy_train_epoch = 0
  best_tree_in_epoch = 0
  best_tree_in = 0
  best_deriv_fscore_epoch = 0
  best_deriv_fscore = FScore()

  for epoch in range(epochs):
    error_log.append("epoch " + str(epoch) + "\n")
    train_start = time.time()
    total_train_loss = 0
    token_accuracy_train = tf.keras.metrics.Accuracy()
    exact_accuracy_train = tf.keras.metrics.Accuracy()
    edge_accuracy_train = tf.keras.metrics.Accuracy()
    train_edge_p = 0
    train_edge_r = 0
    train_edge_f1 = 0
    for batch, examples in enumerate(train_set):
      batch_loss, _, _, p, r, f1 = process_one_batch(
          model,
          sequence_loss_fn,
          edge_loss_fn,
          examples,
          tgt_vocab,
          optimizer,
          token_accuracy_train,
          exact_accuracy_train,
          edge_accuracy=edge_accuracy_train)
      total_train_loss += batch_loss
      train_edge_p += p
      train_edge_r += r
      train_edge_f1 += f1
      if batch % 100 == 0:
        logging.info("Epoch {} Batch {} Loss {:.4f}".format(
            epoch + 1, batch, batch_loss.numpy()))

    num_train_batch = batch + 1
    train_edge_p /= num_train_batch
    train_edge_r /= num_train_batch
    train_edge_f1 /= num_train_batch

    token_accuracy_train_val = token_accuracy_train.result().numpy()
    exact_accuracy_train_val = exact_accuracy_train.result().numpy()
    edge_accuracy_train_val = edge_accuracy_train.result().numpy()
    if best_token_accuracy_train < token_accuracy_train_val:
      best_token_accuracy_train = token_accuracy_train_val
      best_token_accuracy_train_epoch = epoch + 1

    dev_start = time.time()

    dev_results = eval_on_dataset(model,
                                  sequence_loss_fn,
                                  edge_loss_fn,
                                  json.load(open(FLAGS.ref_derivs)),
                                  dev_set,
                                  src_vocab,
                                  tgt_vocab,
                                  calc_deriv_metrics=True)

    dev_tree_in = dev_results["naive_tree_in"]
    deriv_fscore = dev_results["fscore"]
    num_dev_batch = batch + 1

    write_model = False

    if deriv_fscore > best_deriv_fscore:
      best_deriv_fscore = deriv_fscore
      best_deriv_fscore_epoch = epoch + 1
      write_model = True
    elif dev_tree_in > best_tree_in:
      best_tree_in = dev_tree_in
      best_tree_in_epoch = epoch + 1
      write_model = True

    message = []
    message.extend([
        "Epoch {} Train Loss {:.4f} TokenAcc {:.4f}".format(
            epoch + 1, total_train_loss / num_train_batch,
            token_accuracy_train_val),
        "EdgeAcc {:.4f} EdgePrecision {:.4f} EdgeRecall {:.4f} EdgeF1 {:.4f}".
        format(edge_accuracy_train_val, train_edge_p, train_edge_r,
               train_edge_f1),
        " ExactMatch {:.4f}".format(exact_accuracy_train_val)
    ])
    message.append("best {:.4f}@{}".format(best_token_accuracy_train,
                                           best_token_accuracy_train_epoch))

    message.append("DerivFScore {}".format(deriv_fscore))
    message.append("best {}@{}".format(best_deriv_fscore,
                                       best_deriv_fscore_epoch))
    message.append("TreeIn {:.4f}".format(dev_tree_in))
    message.append("best {:.4f}@{}".format(best_tree_in, best_tree_in_epoch))
    message = " ".join(message)

    logging.info(message)
    if save_model_path:
      with open(os.path.join(save_model_path, "log"), "a") as log_f:
        print(message, file=log_f)
    if write_model:
      if save_model_path:
        src_vocab.save(os.path.join(save_model_path, "src_vocab"))
        tgt_vocab.save(os.path.join(save_model_path, "tgt_vocab"))
        json.dump(model.hparams,
                  open(os.path.join(save_model_path, "hparams.json"), "w"))
        model.save_weights(os.path.join(save_model_path, "model_weights"),
                           save_format="tf")
        with open(os.path.join(save_model_path, "stats"), "w") as stats_f:
          print(message, file=stats_f)
        logging.info("Model saved to {}.".format(save_model_path))

    logging.info("  Time taken for 1 epoch train {}s dev {}s\n".format(
        dev_start - train_start, dev_results["time"]))


def eval_on_dataset(model,
                    sequence_loss_fn,
                    edge_loss_fn,
                    all_ref_derivs,
                    dataset,
                    src_vocab,
                    tgt_vocab,
                    calc_deriv_metrics=True):
  dev_start = time.time()
  total_loss = 0
  token_accuracy = tf.keras.metrics.Accuracy()
  exact_accuracy = tf.keras.metrics.Accuracy()
  edge_accuracy = tf.keras.metrics.Accuracy()
  results = []
  for batch, examples in enumerate(dataset):
    (batch_loss, predictions, predictions_edge, _, _,
     _) = process_one_batch(model,
                            sequence_loss_fn,
                            edge_loss_fn,
                            examples,
                            tgt_vocab,
                            None,
                            token_accuracy,
                            exact_accuracy,
                            is_train=False,
                            edge_accuracy=edge_accuracy)
    results.append((dict((k, v.numpy()) for k, v in examples.items()),
                    predictions.numpy(), predictions_edge.numpy()))
    total_loss += batch_loss
  num_batch = batch + 1

  token_accuracy_val = token_accuracy.result().numpy()
  exact_accuracy_val = exact_accuracy.result().numpy()
  edge_accuracy_val = edge_accuracy.result().numpy()

  result_dicts = []
  for examples, predictions, predictions_edge in results:
    batch_sz = examples["src_token_ids"].shape[0]
    batch_dicts = [dict() for _ in range(batch_sz)]
    for key, tensor in examples.items():
      for i in range(batch_sz):
        batch_dicts[i][key] = tensor[i].tolist()
    for i in range(batch_sz):
      batch_dicts[i]["predicted_tgt_token_ids"] = predictions[i].tolist()
      if tgt_vocab.token2idx[
          tgt_vocab.EOS] in batch_dicts[i]["predicted_tgt_token_ids"]:
        predicted_tgt_length = batch_dicts[i]["predicted_tgt_token_ids"].index(
            tgt_vocab.token2idx[tgt_vocab.EOS]) + 1
      else:
        predicted_tgt_length = len(batch_dicts[i]["predicted_tgt_token_ids"])
      batch_dicts[i]["predicted_tgt_edges"] = predictions_edge[i]
    result_dicts.extend(batch_dicts)

  if len(result_dicts) != len(all_ref_derivs):
    logging.error("Mismatch results and ref derivs: {} vs. {}".format(
        len(result_dicts), len(all_ref_derivs)))

  contains_gold_tree_count = 0
  naive_tree_in_count = 0
  all_deriv_fscore = FScore()

  for exid, (example, ref_derivs) in enumerate(zip(result_dicts,
                                                   all_ref_derivs)):
    src_token_ids = example["src_token_ids"]
    src_tokens = [src_vocab.idx2token[idx] for idx in src_token_ids]
    example["src_tokens"] = src_tokens

    tgt_token_ids = example["tgt_token_ids"]
    tgt_tokens = [
        tgt_vocab.idx2token[idx]
        if idx < len(tgt_vocab) else src_tokens[idx - len(tgt_vocab)]
        for idx in tgt_token_ids
    ]
    tgt_length = tgt_token_ids.index(tgt_vocab.token2idx[tgt_vocab.EOS]) + 1
    tgt_edges = np.array(
        example["tgt_edges"])[:tgt_length, :tgt_length].reshape(
            (tgt_length, tgt_length))
    tgt_tree = reconstruct_tree(tgt_token_ids[1:], tgt_edges[1:],
                                tgt_length - 2, tgt_vocab)
    if not tgt_tree:
      logging.error("Invalid ref dag")
      continue

    predicted_tgt_token_ids = example["predicted_tgt_token_ids"]
    predicted_tgt_tokens = [
        tgt_vocab.idx2token[idx]
        if idx < len(tgt_vocab) else src_tokens[idx - len(tgt_vocab)]
        for idx in predicted_tgt_token_ids
    ]
    if tgt_vocab.token2idx[tgt_vocab.EOS] in predicted_tgt_token_ids:
      predicted_tgt_length = predicted_tgt_token_ids.index(
          tgt_vocab.token2idx[tgt_vocab.EOS])
    else:
      predicted_tgt_length = len(predicted_tgt_token_ids) - 1
    predicted_tgt_edges = (example["predicted_tgt_edges"]
                           [:predicted_tgt_length, :predicted_tgt_length])

    assert predicted_tgt_token_ids[0] == tgt_vocab.token2idx[tgt_vocab.BOS]
    predict_tree = reconstruct_tree(predicted_tgt_token_ids[1:],
                                    predicted_tgt_edges,
                                    predicted_tgt_length - 1,
                                    tgt_vocab,
                                    robust_mode=True)

    if predict_tree:
      if contains_tree(predict_tree[0], tgt_tree[0]):
        naive_tree_in_count += 1

      if not calc_deriv_metrics:
        continue

      children_comb = precompute_children_combinations(predict_tree[0])
      predicted_trees = []
      for t in retrieve_trees(predict_tree[0]):
        predicted_trees.append([n for n in t if n != tgt_vocab.BOS])
      predicted_tree_strings = [" ".join(tree) for tree in predicted_trees]
      if ref_derivs[0] in predicted_tree_strings:
        contains_gold_tree_count += 1
      ref_tree_set = set(ref_derivs)
      predicted_tree_set = set(predicted_tree_strings)
      deriv_fscore = FScore(correct=len(predicted_tree_set & ref_tree_set),
                            predcount=len(predicted_tree_set),
                            goldcount=len(ref_tree_set))
      all_deriv_fscore += deriv_fscore
  return {
      "predictions": result_dicts,
      "naive_tree_in": float(naive_tree_in_count) / len(result_dicts),
      "tree_in": float(contains_gold_tree_count) / len(result_dicts),
      "fscore": all_deriv_fscore,
      "time": time.time() - dev_start,
      "token_accuracy": token_accuracy_val,
      "exact_accuracy": exact_accuracy_val,
      "edge_accuracy": edge_accuracy_val,
      "loss": total_loss / num_batch
  }


def predict(model_type, eval_set, src_vocab, tgt_vocab, save_model_path,
            all_ref_derivs, predict_output):
  hparams = json.load(open(os.path.join(save_model_path, "hparams.json")))
  model = model_type(src_vocab, tgt_vocab, hparams)
  model.load_weights(os.path.join(save_model_path, "model_weights"))
  sequence_loss_fn = SequenceLoss(tgt_vocab)
  edge_loss_fn = EdgeLoss()

  results = eval_on_dataset(model, sequence_loss_fn, edge_loss_fn,
                            all_ref_derivs, eval_set, src_vocab, tgt_vocab)

  logging.info(
      "Loss {:.4f} TokenAcc {:.4f} EdgeAcc {:.4f} ExactMatch {:.4f}".format(
          results["loss"], results["token_accuracy"], results["edge_accuracy"],
          results["exact_accuracy"]))
  logging.info("NaiveTreeIn {:.4f} GoldTreeIn {:.4f} DerivFScore {}".format(
      results["naive_tree_in"], results["tree_in"], results["fscore"]))

  logging.info("  Time taken: {}s\n".format(results["time"]))

  if predict_output:
    # Rebuild the tokens
    with open(os.path.join(predict_output), "w") as output_f:
      for exid, example in enumerate(results["predictions"]):
        src_token_ids = example["src_token_ids"]
        src_tokens = example["src_tokens"]
        tgt_token_ids = example["tgt_token_ids"]
        tgt_tokens = [
            tgt_vocab.idx2token[idx]
            if idx < len(tgt_vocab) else src_tokens[idx - len(tgt_vocab)]
            for idx in tgt_token_ids
        ]
        predicted_tgt_token_ids = example["predicted_tgt_token_ids"]
        predicted_tgt_tokens = [
            tgt_vocab.idx2token[idx]
            if idx < len(tgt_vocab) else src_tokens[idx - len(tgt_vocab)]
            for idx in predicted_tgt_token_ids
        ]
        tgt_length = tgt_token_ids.index(tgt_vocab.token2idx[tgt_vocab.EOS]) + 1
        tgt_edges = np.array(
            example["tgt_edges"])[:tgt_length, :tgt_length].reshape(
                (tgt_length, tgt_length))
        if tgt_vocab.token2idx[tgt_vocab.EOS] in predicted_tgt_token_ids:
          predicted_tgt_length = predicted_tgt_token_ids.index(
              tgt_vocab.token2idx[tgt_vocab.EOS])
        else:
          predicted_tgt_length = len(predicted_tgt_token_ids) - 1
        predicted_tgt_edges = (example["predicted_tgt_edges"]
                               [:predicted_tgt_length, :predicted_tgt_length])

        print("example", exid, file=output_f)
        print("src_token_ids:\t",
              ", ".join(str(v) for v in src_token_ids),
              file=output_f)
        print("src_tokens:\t", " ".join(src_tokens), file=output_f)
        print("tgt_token_ids:\t",
              ", ".join(str(v) for v in tgt_token_ids),
              file=output_f)
        print("tgt_tokens:\t", " ".join(tgt_tokens), file=output_f)
        print("predicted_tgt_token_ids:\t",
              ", ".join(str(v) for v in predicted_tgt_token_ids),
              file=output_f)
        print("predicted_tgt_tokens:\t",
              " ".join(predicted_tgt_tokens),
              file=output_f)
        print("tgt edges", file=output_f)
        print(tgt_edges, file=output_f)
        print("predicted tgt edges", file=output_f)
        print(predicted_tgt_edges, file=output_f)
        print("\n", file=output_f)
    print("Deriv FScore", results["fscore"])


def main(argv):
  del argv  # Unused.

  if FLAGS.eager_run:
    tf.config.experimental_run_functions_eagerly(True)

  model_type = GraphTransformer

  data_spec = json.load(open(FLAGS.data_spec))

  if FLAGS.predict:
    src_vocab = Vocabulary.load(data_spec["source_vocab"])
    tgt_vocab = Vocabulary.load(data_spec["target_vocab"])
    eval_set = build_dataset(os.path.join(FLAGS.predict),
                             max_num_src_tokens=data_spec["max_num_src_tokens"],
                             max_num_tgt_tokens=data_spec["max_num_tgt_tokens"],
                             src_vocab=src_vocab,
                             tgt_vocab=tgt_vocab,
                             is_train=False,
                             predict_edge=True,
                             multiple=True)
    eval_set = eval_set.batch(FLAGS.batch_size, drop_remainder=False)
    predict(model_type, eval_set, src_vocab, tgt_vocab, FLAGS.save_model_path,
            json.load(open(FLAGS.ref_derivs)), FLAGS.predict_output)
  else:
    hparams = {
        "batch_sz": FLAGS.batch_size,
        "d_model": FLAGS.model_dim,
        "biaffine": FLAGS.biaffine,
        "num_src_tokens": data_spec["max_num_src_tokens"],
        "num_tgt_tokens": data_spec["max_num_tgt_tokens"],
        "dropout": FLAGS.dropout,
        "predict_edge": True
    }
    src_vocab = Vocabulary.load(data_spec["source_vocab"])
    tgt_vocab = Vocabulary.load(data_spec["target_vocab"])
    train_set = build_dataset(data_spec["train_set"],
                              max_num_src_tokens=hparams["num_src_tokens"],
                              max_num_tgt_tokens=hparams["num_tgt_tokens"],
                              src_vocab=src_vocab,
                              tgt_vocab=tgt_vocab,
                              is_train=True,
                              predict_edge=True,
                              multiple=True)
    train_set = train_set.batch(FLAGS.batch_size, drop_remainder=True)
    dev_set = build_dataset(data_spec["dev_set"],
                            max_num_src_tokens=hparams["num_src_tokens"],
                            max_num_tgt_tokens=hparams["num_tgt_tokens"],
                            src_vocab=src_vocab,
                            tgt_vocab=tgt_vocab,
                            is_train=False,
                            predict_edge=True,
                            multiple=True)
    dev_set = dev_set.batch(FLAGS.batch_size, drop_remainder=False)

    train(model_type, FLAGS.epochs, train_set, dev_set, src_vocab, tgt_vocab,
          hparams, FLAGS.save_model_path)


if __name__ == "__main__":
  app.run(main)
