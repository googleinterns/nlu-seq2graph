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
"""FScore."""


class FScore(object):
  """FScore."""

  def __init__(self, correct=0, predcount=0, goldcount=0):
    self.correct = correct  # correct brackets
    self.predcount = predcount  # total predicted brackets
    self.goldcount = goldcount  # total gold brackets

  def precision(self):
    if self.predcount > 0:
      return (100.0 * self.correct) / self.predcount
    else:
      return 0.0

  def recall(self):
    if self.goldcount > 0:
      return (100.0 * self.correct) / self.goldcount
    else:
      return 0.0

  def fscore(self):
    precision = self.precision()
    recall = self.recall()
    if (precision + recall) > 0:
      return (2 * precision * recall) / (precision + recall)
    else:
      return 0.0

  def __str__(self):
    precision = self.precision()
    recall = self.recall()
    fscore = self.fscore()
    return '(P= {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
        precision,
        recall,
        fscore,
    )

  def __repr__(self):
    return str(self)

  def __iadd__(self, other):
    self.correct += other.correct
    self.predcount += other.predcount
    self.goldcount += other.goldcount
    return self

  def __add__(self, other):
    return FScore(self.correct + other.correct,
                  self.predcount + other.predcount,
                  self.goldcount + other.goldcount)

  def __lt__(self, other):
    if isinstance(other, FScore):
      return self.fscore() < other.fscore()
    else:
      return self.fscore < other

  def __gt__(self, other):
    if isinstance(other, FScore):
      return self.fscore() > other.fscore()
    else:
      return self.fscore > other

  def detailed_str(self):
    return ('(Pred= {}, Gold={}, Correct={})'.format(
        self.predcount, self.goldcount, self.correct) + '\t' + self.__str__())
