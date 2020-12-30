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

import os
import json
import sys
import shutil
import time

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from vocabulary import Vocabulary
from dataset import build_dataset
from graph_transformer import GraphTransformer
from graph_utils import contain_tree
from graph_utils import reconstruct_tree
from training_utils import NoamSchedule
from training_utils import EdgeLoss
from training_utils import SequenceLoss

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
flags.DEFINE_bool("test_seq2graph", False,
                  "Use seq2graph metric in evaluating dev set.")

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

    # Shape: [batch_sz, max_num_tgt_tokens]
    if not FLAGS.test_seq2graph or (FLAGS.test_seq2graph and is_train):

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
  if not FLAGS.test_seq2graph:
    best_token_accuracy_dev = 0
    best_token_accuracy_dev_epoch = 0
  else:
    best_tree_in = 0
    best_tree_in_epoch = 0

  for epoch in range(epochs):
    error_log.append('epoch ' + str(epoch) + '\n')
    train_start = time.time()
    total_train_loss = 0
    token_accuracy_train = tf.keras.metrics.Accuracy()
    exact_accuracy_train = tf.keras.metrics.Accuracy()
    edge_accuracy_train = tf.keras.metrics.Accuracy()
    train_p = 0
    train_r = 0
    train_f1 = 0
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
      train_p += p
      train_r += r
      train_f1 += f1
      if batch % 100 == 0:
        print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch,
                                                     batch_loss.numpy()))

    num_train_batch = batch + 1
    train_p /= num_train_batch
    train_r /= num_train_batch
    train_f1 /= num_train_batch

    dev_start = time.time()
    total_dev_loss = 0
    token_accuracy_dev = tf.keras.metrics.Accuracy()
    exact_accuracy_dev = tf.keras.metrics.Accuracy()
    edge_accuracy_dev = tf.keras.metrics.Accuracy()
    if not FLAGS.test_seq2graph:
      dev_p = 0
      dev_r = 0
      dev_f1 = 0
    else:
      tree_in = 0
      all_tree = 0
      cycle_num = 0
    for batch, examples in enumerate(dev_set):
      (batch_loss, dev_predictions, dev_prediction_edge, p, r,
       f1) = process_one_batch(model,
                               sequence_loss_fn,
                               edge_loss_fn,
                               examples,
                               tgt_vocab,
                               optimizer,
                               token_accuracy_dev,
                               exact_accuracy_dev,
                               is_train=False,
                               edge_accuracy=edge_accuracy_dev)
      total_dev_loss += batch_loss
      if not FLAGS.test_seq2graph:
        dev_p += p
        dev_r += r
        dev_f1 += f1
      else:
        for e in range(len(examples["tgt_token_ids"])):

          orig_tgt_token_ids = examples["tgt_token_ids"][e].numpy().tolist()
          orig_tgt_edges = examples["tgt_edges"][e].numpy().tolist()
          orig_length = sum(tok != tgt_vocab.token2idx[tgt_vocab.PAD]
                            for tok in orig_tgt_token_ids)

          dev_pred = dev_predictions[e].numpy().tolist()
          dev_edge = dev_prediction_edge[e].numpy().tolist()
          if tgt_vocab.token2idx[tgt_vocab.EOS] in dev_pred:
            dev_length = dev_pred.index(tgt_vocab.token2idx[tgt_vocab.EOS])
          else:
            dev_length = len(dev_pred) - 1

          orig_tree, _ = reconstruct_tree(orig_tgt_token_ids, orig_tgt_edges,
                                          orig_length, tgt_vocab)
          predict_tree, has_cycle = reconstruct_tree(dev_pred[1:],
                                                     dev_edge[:-1],
                                                     dev_length,
                                                     tgt_vocab,
                                                     with_bos=True)

          all_tree += 1

          if contain_tree(predict_tree[0], orig_tree[0]):
            tree_in += 1
          else:
            src_token_ids = examples["src_token_ids"][e].numpy().tolist()
            src_length = sum(tok != src_vocab.token2idx[src_vocab.PAD]
                             for tok in src_token_ids)
            src_tokens = [
                src_vocab.idx2token[idx] for idx in src_token_ids[:src_length]
            ]
            error_log.append('src tokens: ' + ' '.join(src_tokens) + '\n')

            tgt_tokens = [
                tgt_vocab.idx2token[idx]
                if idx < len(tgt_vocab) else str(idx - len(tgt_vocab))
                for idx in orig_tgt_token_ids[:orig_length]
            ]
            error_log.append('tgt tokens: ' + ' '.join(tgt_tokens) + '\n')

            error_log.append('orig edge:\n')

            for ss in range(len(orig_tgt_edges[:orig_length])):
              error_log.append(' '.join(
                  [str(tt) for tt in orig_tgt_edges[ss][:orig_length]]) + '\n')
            error_log.append('orig tree: ' +
                             ' '.join([v.word for v in orig_tree]) + '\n')

            pred_tokens = [
                tgt_vocab.idx2token[idx]
                if idx < len(tgt_vocab) else str(idx - len(tgt_vocab))
                for idx in dev_pred[:dev_length + 1]
            ]
            error_log.append('pred tokens: ' + ' '.join(pred_tokens) + '\n')
            error_log.append('pred edge:\n')

            for ss in range(len(dev_edge[:dev_length])):
              error_log.append(
                  ' '.join([str(tt) for tt in dev_edge[ss][:dev_length + 1]]) +
                  '\n')
            error_log.append('pred tree: ' +
                             ' '.join([v.word for v in predict_tree]) + '\n')
          if has_cycle:
            cycle_num += 1

            pred_tokens = [
                tgt_vocab.idx2token[idx]
                if idx < len(tgt_vocab) else str(idx - len(tgt_vocab))
                for idx in dev_pred[:dev_length + 1]
            ]
            print('pred tokens: ' + ' '.join(pred_tokens))
            print('pred edge:')

    if log_error:
      with open(os.path.join(save_model_path, 'error_log'), "a") as log_error:
        print(''.join(error_log), file=log_error)

    if FLAGS.test_seq2graph:
      dev_tree_in = tree_in / (all_tree * 1.0)
    num_dev_batch = batch + 1
    if not FLAGS.test_seq2graph:
      dev_p /= num_dev_batch
      dev_r /= num_dev_batch
      dev_f1 /= num_dev_batch

    epoch_finish = time.time()

    token_accuracy_train_val = token_accuracy_train.result().numpy()
    exact_accuracy_train_val = exact_accuracy_train.result().numpy()
    edge_accuracy_train_val = edge_accuracy_train.result().numpy()
    if not FLAGS.test_seq2graph:
      token_accuracy_dev_val = token_accuracy_dev.result().numpy()
      exact_accuracy_dev_val = exact_accuracy_dev.result().numpy()
      edge_accuracy_dev_val = edge_accuracy_dev.result().numpy()
    if best_token_accuracy_train < token_accuracy_train_val:
      best_token_accuracy_train = token_accuracy_train_val
      best_token_accuracy_train_epoch = epoch + 1
    write_model = False

    if not FLAGS.test_seq2graph:
      if best_token_accuracy_dev < token_accuracy_dev_val:
        best_token_accuracy_dev = token_accuracy_dev_val
        best_token_accuracy_dev_epoch = epoch + 1
        write_model = True
    else:
      if dev_tree_in > best_tree_in:
        best_tree_in = dev_tree_in
        best_tree_in_epoch = epoch + 1
        write_model = True

    message = []
    message.append(
        "Epoch {} Train Loss {:.4f} TokenAcc {:.4f} EdgeAcc {:.4f} EdgePrecision {:.4f} EdgeRecall {:.4f} EdgeF1 {:.4f} ExactMatch {:.4f}"
        .format(epoch + 1, total_train_loss / num_train_batch,
                token_accuracy_train_val, edge_accuracy_train_val, train_p,
                train_r, train_f1, exact_accuracy_train_val))
    message.append("best {:.4f}@{}".format(best_token_accuracy_train,
                                           best_token_accuracy_train_epoch))

    if not FLAGS.test_seq2graph:
      message.append(
          "Dev Loss {:.4f} TokenAcc {:.4f} EdgeAcc {:.4f} EdgePrecision {:.4f} EdgeRecall {:.4f} EdgeF1 {:.4f} ExactMatch {:.4f}"
          .format(total_dev_loss / num_dev_batch, token_accuracy_dev_val,
                  edge_accuracy_dev_val, dev_p, dev_r, dev_f1,
                  exact_accuracy_dev_val))
      message.append("best {:.4f}@{}".format(best_token_accuracy_dev,
                                             best_token_accuracy_dev_epoch))

    else:
      message.append("TreeIn {:.4f}".format(dev_tree_in))
      message.append("best {:.4f}@{}".format(best_tree_in, best_tree_in_epoch))
      message.append("cycle size: " + str(cycle_num))
    message = " ".join(message)

    print(message)
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
        print("Model saved to {}.".format(save_model_path))

    print("  Time taken for 1 epoch train {}s dev {}s\n".format(
        dev_start - train_start, epoch_finish - dev_start))


def predict(model_type, eval_set, src_vocab, tgt_vocab, save_model_path,
            predict_output):
  hparams = json.load(open(os.path.join(save_model_path, "hparams.json")))
  hparams["predict_edge"] = False
  model = model_type(src_vocab, tgt_vocab, hparams)
  model.load_weights(os.path.join(save_model_path, "model_weights"))
  sequence_loss_fn = SequenceLoss(tgt_vocab)
  edge_loss_fn = EdgeLoss()

  dev_start = time.time()
  total_loss = 0
  token_accuracy = tf.keras.metrics.Accuracy()
  exact_accuracy = tf.keras.metrics.Accuracy()
  edge_accuracy = tf.keras.metrics.Accuracy()
  results = []
  for batch, examples in enumerate(eval_set):
    (batch_loss, predictions, predictions_edge, _, _,
     _) = process_one_batch(model,
                            sequence_loss_fn, edge_loss_fn,
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
  print("Loss {:.4f} TokenAcc {:.4f} EdgeAcc {:.4f} ExactMatch {:.4f}".format(
      total_loss / num_batch, token_accuracy_val, edge_accuracy_val,
      exact_accuracy_val))

  print("  Time taken: {}s\n".format(time.time() - dev_start))

  if predict_output:
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

    # Rebuild the tokens
    with open(os.path.join(predict_output), "w") as output_f:
      for example in result_dicts:
        src_token_ids = example["src_token_ids"]
        src_tokens = [src_vocab.idx2token[idx] for idx in src_token_ids]
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

        tgt_tree, _ = reconstruct_tree(tgt_token_ids, tgt_edges, tgt_length - 1,
                                       tgt_vocab)
        predict_tree, _ = reconstruct_tree(predicted_tgt_token_ids[1:],
                                           predicted_tgt_edges[:-1],
                                           predicted_tgt_length - 1,
                                           tgt_vocab,
                                           with_bos=True)

        tree_in = contain_tree(predict_tree[0], tgt_tree[0])

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
        print("!!! tree in", tree_in, file=output_f)
        print("\n", file=output_f)


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
            FLAGS.predict_output)
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
    dev_set = dev_set.batch(FLAGS.batch_size, drop_remainder=True)

    train(model_type, FLAGS.epochs, train_set, dev_set, src_vocab, tgt_vocab,
          hparams, FLAGS.save_model_path)


if __name__ == "__main__":
  app.run(main)
