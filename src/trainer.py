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
  --predict ~/workspace/seq2graph/spider/train.sstable \
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

from vocabulary import Vocabulary
from dataset import build_dataset
from transformer import Transformer

flags.DEFINE_string("data_spec", None, "Path to training data spec.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("model_dim", 128, "Model dim.")
flags.DEFINE_integer("epochs", 10, "Num of epochs.")
flags.DEFINE_integer("beam_size", 1, "beam size.")
flags.DEFINE_float("dropout", 0.2, "Dropout rate.")
flags.DEFINE_string("save_model_path", None, "Save model path.")
flags.DEFINE_string(
    "predict", None,
    "Init model from save_model_path and run prediction on the data set,")
flags.DEFINE_string("predict_output", None, "Prediction output.")
flags.DEFINE_bool("eager_run", False, "Run in eager mode for debugging.")

FLAGS = flags.FLAGS


class SequenceLoss(object):

  def __init__(self, tgt_vocab):
    self.tgt_pad_id = tgt_vocab.token2idx[tgt_vocab.PAD]
    self.tgt_vocab_size = len(tgt_vocab)
    self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none")

  def __call__(self, pred, ref_token_ids):
    mask = tf.math.logical_not(tf.math.equal(ref_token_ids, self.tgt_pad_id))
    loss = self.loss_fn(ref_token_ids, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)


@tf.function
def process_one_batch(model,
                      loss_fn,
                      examples,
                      tgt_vocab,
                      optimizer,
                      token_accuracy,
                      exact_accuracy,
                      is_train=True):
  with tf.GradientTape() as tape:
    # Shape: [batch_sz, max_num_tgt_tokens, tgt_vocab_size + max_num_src_tokens]
    tgt_token_ids = examples["tgt_token_ids"]
    if FLAGS.beam_size > 1 and is_train is False:
      beam_predictions, scores = model(examples,
                                       beam_size=FLAGS.beam_size,
                                       is_train=is_train)

      eos_index = tf.cast(tf.equal(beam_predictions,
                                   tgt_vocab.token2idx[tgt_vocab.EOS]),
                          dtype=tf.int32)
      last_index = tf.cast(tf.equal(tf.range(beam_predictions.shape[2]),
                                    beam_predictions.shape[2] - 1),
                           dtype=tf.int32)
      non_eos_index = 1 - tf.reduce_sum(eos_index, axis=-1)
      non_eos_index = tf.expand_dims(non_eos_index, 2) * tf.expand_dims(
          last_index, 0)
      eos_index += non_eos_index
      eos_scores = tf.reshape(
          tf.boolean_mask(scores, tf.cast(eos_index, dtype=tf.bool)),
          [-1, FLAGS.beam_size])
      _, best_scores_index = tf.math.top_k(eos_scores, k=1)
      predictions = tf.cast(tf.squeeze(
          tf.gather_nd(beam_predictions,
                       tf.expand_dims(best_scores_index, 2),
                       batch_dims=1), 1),
                            dtype=tf.int64)
      loss = 0

    else:
      prediction_logits = model(examples, is_train=is_train)
      loss = loss_fn(prediction_logits, tgt_token_ids)

      predictions = tf.argmax(prediction_logits, axis=-1)

    if is_train:
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Shape: [batch_sz, max_num_tgt_tokens]
    oov_mask = tf.not_equal(tgt_token_ids, tgt_vocab.token2idx[tgt_vocab.PAD])
    token_accuracy.update_state(tgt_token_ids,
                                predictions,
                                sample_weight=tf.cast(oov_mask,
                                                      dtype=tf.float32))

    exact_match = tf.cast(
        tf.reduce_all(tf.logical_or(tf.equal(tgt_token_ids, predictions),
                                    tf.logical_not(oov_mask)),
                      axis=1), tf.float32)

    exact_accuracy.update_state(tf.ones_like(exact_match), exact_match)

  return loss, predictions


def train(model_type, epochs, train_set, dev_set, src_vocab, tgt_vocab, hparams,
          save_model_path):
  optimizer = tf.keras.optimizers.Adam()
  model = model_type(src_vocab, tgt_vocab, hparams)
  loss_fn = SequenceLoss(tgt_vocab)

  if save_model_path:
    try:
      shutil.rmtree(save_model_path)
    except:
      pass
    os.mkdir(save_model_path)

  best_token_accuracy_train = 0
  best_token_accuracy_train_epoch = 0
  best_token_accuracy_dev = 0
  best_token_accuracy_dev_epoch = 0

  for epoch in range(epochs):
    train_start = time.time()
    total_train_loss = 0
    token_accuracy_train = tf.keras.metrics.Accuracy()
    exact_accuracy_train = tf.keras.metrics.Accuracy()
    for batch, examples in enumerate(train_set):
      batch_loss, _ = process_one_batch(model,
                                        loss_fn,
                                        examples,
                                        tgt_vocab,
                                        optimizer,
                                        token_accuracy_train,
                                        exact_accuracy_train)
      total_train_loss += batch_loss

      if batch % 100 == 0:
        print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch,
                                                     batch_loss.numpy()))

    num_train_batch = batch + 1

    dev_start = time.time()
    total_dev_loss = 0
    token_accuracy_dev = tf.keras.metrics.Accuracy()
    exact_accuracy_dev = tf.keras.metrics.Accuracy()
    for batch, examples in enumerate(dev_set):
      batch_loss, _ = process_one_batch(model,
                                        loss_fn,
                                        examples,
                                        tgt_vocab,
                                        optimizer,
                                        token_accuracy_dev,
                                        exact_accuracy_dev,
                                        is_train=False)

      total_dev_loss += batch_loss
      # break
    num_dev_batch = batch + 1

    epoch_finish = time.time()

    token_accuracy_train_val = token_accuracy_train.result().numpy()
    exact_accuracy_train_val = exact_accuracy_train.result().numpy()
    token_accuracy_dev_val = token_accuracy_dev.result().numpy()
    exact_accuracy_dev_val = exact_accuracy_dev.result().numpy()
    if best_token_accuracy_train < token_accuracy_train_val:
      best_token_accuracy_train = token_accuracy_train_val
      best_token_accuracy_train_epoch = epoch + 1
    write_model = False
    if best_token_accuracy_dev < token_accuracy_dev_val:
      best_token_accuracy_dev = token_accuracy_dev_val
      best_token_accuracy_dev_epoch = epoch + 1
      write_model = True

    message = " ".join([
        "Epoch {} Train Loss {:.4f} TokenAcc {:.4f} ExactMatch {:.4f}".format(
            epoch + 1, total_train_loss / num_train_batch,
            token_accuracy_train_val, exact_accuracy_train_val),
        "best {:.4f}@{}".format(best_token_accuracy_train,
                                best_token_accuracy_train_epoch),
        "Dev Loss {:.4f} TokenAcc {:.4f} ExactMatch {:.4f}".format(
            total_dev_loss / num_dev_batch, token_accuracy_dev_val,
            exact_accuracy_dev_val),
        "best {:.4f}@{}".format(best_token_accuracy_dev,
                                best_token_accuracy_dev_epoch)
    ])
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
  model = model_type(src_vocab, tgt_vocab, hparams)
  model.load_weights(os.path.join(save_model_path, "model_weights"))
  loss_fn = SequenceLoss(tgt_vocab)

  dev_start = time.time()
  total_loss = 0
  token_accuracy = tf.keras.metrics.Accuracy()
  exact_accuracy = tf.keras.metrics.Accuracy()
  results = []
  for batch, examples in enumerate(eval_set):
    tgt_token_ids = examples["tgt_token_ids"]

    batch_loss, predictions = process_one_batch(model,
                                                loss_fn,
                                                examples,
                                                tgt_vocab,
                                                None,
                                                token_accuracy,
                                                exact_accuracy,
                                                is_train=False)

    results.append((dict(
        (k, v.numpy()) for k, v in examples.items()), predictions.numpy()))
    total_loss += batch_loss
  num_batch = batch + 1

  token_accuracy_val = token_accuracy.result().numpy()
  exact_accuracy_val = exact_accuracy.result().numpy()
  print("Loss {:.4f} TokenAcc {:.4f} ExactMatch {:.4f}".format(
      total_loss / num_batch, token_accuracy_val, exact_accuracy_val))
  print("  Time taken: {}s\n".format(time.time() - dev_start))

  if predict_output:
    result_dicts = []
    for examples, predictions in results:
      batch_sz = examples["src_token_ids"].shape[0]
      batch_dicts = [dict() for _ in range(batch_sz)]
      for key, tensor in examples.items():
        for i in range(batch_sz):
          batch_dicts[i][key] = tensor[i].tolist()
      for i in range(batch_sz):
        batch_dicts[i]["predicted_tgt_token_ids"] = predictions[i].tolist()
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
        print("\n", file=output_f)


def main(argv):
  del argv  # Unused.

  if FLAGS.eager_run:
    tf.config.experimental_run_functions_eagerly(True)

  model_type = Transformer

  data_spec = json.load(open(FLAGS.data_spec))

  if FLAGS.predict:
    src_vocab = Vocabulary.load(data_spec["source_vocab"])
    tgt_vocab = Vocabulary.load(data_spec["target_vocab"])
    eval_set = build_dataset(os.path.join(FLAGS.predict),
                             max_num_src_tokens=data_spec["max_num_src_tokens"],
                             max_num_tgt_tokens=data_spec["max_num_tgt_tokens"],
                             src_vocab=src_vocab,
                             tgt_vocab=tgt_vocab,
                             is_train=False)
    eval_set = eval_set.batch(FLAGS.batch_size, drop_remainder=False)
    predict(model_type, eval_set, src_vocab, tgt_vocab, FLAGS.save_model_path,
            FLAGS.predict_output)
  else:
    hparams = {
        "batch_sz": FLAGS.batch_size,
        "d_model": FLAGS.model_dim,
        "num_src_tokens": data_spec["max_num_src_tokens"],
        "num_tgt_tokens": data_spec["max_num_tgt_tokens"],
        "dropout": FLAGS.dropout,
    }
    src_vocab = Vocabulary.load(data_spec["source_vocab"])
    tgt_vocab = Vocabulary.load(data_spec["target_vocab"])
    train_set = build_dataset(data_spec["train_set"],
                              max_num_src_tokens=hparams["num_src_tokens"],
                              max_num_tgt_tokens=hparams["num_tgt_tokens"],
                              src_vocab=src_vocab,
                              tgt_vocab=tgt_vocab,
                              is_train=True)
    train_set = train_set.batch(FLAGS.batch_size, drop_remainder=True)
    dev_set = build_dataset(data_spec["dev_set"],
                            max_num_src_tokens=hparams["num_src_tokens"],
                            max_num_tgt_tokens=hparams["num_tgt_tokens"],
                            src_vocab=src_vocab,
                            tgt_vocab=tgt_vocab,
                            is_train=False)
    dev_set = dev_set.batch(FLAGS.batch_size, drop_remainder=True)

    train(model_type, FLAGS.epochs, train_set, dev_set, src_vocab, tgt_vocab,
          hparams, FLAGS.save_model_path)


if __name__ == "__main__":
  app.run(main)
