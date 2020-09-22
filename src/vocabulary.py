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
"""Vocabulary utilities."""

import collections

from absl import logging


class Vocabulary(object):
  """Maintains the vocabulary."""
  OOV = "<OOV>"
  BOS = "<BOS>"
  EOS = "<EOS>"
  PAD = "<PAD>"
  SEP = "<SEP>"
  # Make sure PAD will be indexed 0.
  meta_tokens = [PAD, OOV, BOS, EOS, SEP]

  def __init__(self):
    self.token2idx = {}
    self.idx2token = []
    self.token2freq = {}

  def add(self, token, freq=-1):
    if token is "":
      return
    if token not in self.token2idx:
      self.token2idx[token] = len(self.token2idx)
      self.idx2token.append(token)
      self.token2freq[token] = freq

  def add_meta_tokens(self):
    for tok in Vocabulary.meta_tokens:
      self.add(tok)

  def convert(self, tokens, add_bos_eos=False):
    """Converts a list of tokens into a list of token indexes."""
    token_indexes = []
    if add_bos_eos:
      token_indexes = [self.token2idx[self.BOS]]
    token_indexes.extend(self.token2idx[token] if token in
                         self.token2idx else self.token2idx[self.OOV]
                         for token in tokens)
    if add_bos_eos:
      token_indexes.append(self.token2idx[self.EOS])
    return token_indexes

  def token_freq(self, token):
    return self.token2freq.get(token, 0)

  def __len__(self):
    return len(self.token2idx)

  def __contains__(self, item):
    return item in self.token2idx

  def save(self, vocab_path):
    with open(vocab_path, "w") as outf:
      for token in self.idx2token:
        print(token, file=outf)

  @staticmethod
  def load(vocab_path):
    vocab = Vocabulary()
    for line in open(vocab_path):
      token = line.strip()
      vocab.token2idx[token] = len(vocab.token2idx)
      vocab.idx2token.append(token)
    return vocab


class VocabularyBuilder(object):
  """Builds a vocabulary."""

  def __init__(self, init_meta_tokens, oov_threshold, max_size=0):
    self.init_meta_tokens = init_meta_tokens
    self.oov_threshold = oov_threshold
    self.max_size = max_size
    self.raw_vocab = collections.Counter()

  def update(self, iterable):
    self.raw_vocab.update(iterable)

  def get_vocabulary(self):
    vocab = Vocabulary()
    vocab_candidate = []
    for tok, count in self.raw_vocab.items():
      if count > self.oov_threshold:
        vocab_candidate.append((count, tok))

    if self.max_size > 0:
      vocab_candidate.sort(reverse=True)
      vocab_candidate = vocab_candidate[:self.max_size]

    if self.init_meta_tokens:
      vocab.add_meta_tokens()

    for count, tok in vocab_candidate:
      vocab.add(tok, count)
    logging.info("%d distinct tokens in total, %d kept", len(self.raw_vocab),
                 len(vocab))
    return vocab
