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
"""Training utilities."""

import tensorflow as tf


class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(NoamSchedule, self).__init__()
    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(
        tf.math.rsqrt(step + 1), (step + 1) * (self.warmup_steps**-1.5))


class EdgeLoss(object):

  def __call__(self, pred, ref, edge_mask):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(
        ref, pred.dtype),
                                                   logits=pred)
    mask = tf.cast(edge_mask, dtype=loss.dtype)
    loss *= mask[:, 1:, tf.newaxis]
    loss *= mask[:, tf.newaxis, :]
    loss = tf.boolean_mask(loss, tf.math.not_equal(loss, 0))
    return tf.reduce_mean(loss)


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
    loss = tf.boolean_mask(loss, tf.math.not_equal(loss, 0))
    return tf.reduce_mean(loss)
