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

from vocabulary import Vocabulary
from vocabulary import VocabularyBuilder


def is_target_token(token):
  return token == ']' or ('[' in token and ':' in token)

def fetch_examples(example_file):
  for example_line in open(example_file):
    try:
      original_query, tokenized_query, reference = example_line.strip().split(
          '\t')
    except ValueError:
      print('Unable to split', example_line)
      continue
    yield (original_query, [tok.lower() for tok in tokenized_query.split()], [
        tok if is_target_token(tok) else tok.lower()
        for tok in reference.split()
    ])


def build_vocab(train_file, src_oov_threshold=1):
  src_vocab_builder = VocabularyBuilder(init_meta_tokens=True,
                                        oov_threshold=src_oov_threshold)
  tgt_vocab_builder = VocabularyBuilder(init_meta_tokens=True, oov_threshold=0)
  for _, src_tokens, ref_tokens in fetch_examples(train_file):
    src_vocab_builder.update(src_tokens)
    for tgt_token in ref_tokens:
      if is_target_token(tgt_token):
        tgt_vocab_builder.update([tgt_token])
      else:
        assert tgt_token in src_tokens
  return (src_vocab_builder.get_vocabulary(),
          tgt_vocab_builder.get_vocabulary())
