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

import numpy as np


class Node(object):

  def __init__(self,
               word,
               parent,
               child=None,
               index=0,
               signature=None,
               identifier=None):
    self.word = word
    self.parent = []
    if parent:
      self.parent.append(parent)
    self.child = []
    if child:
      self.child.append(child)
    self.index = index
    self.signature = [word] if signature is None else signature
    self.identifier = self.word if identifier is None else identifier
    self.sig_set = False
    self.visited = False

  def is_leaf(self):
    return len(self.child) == 0

  def is_root(self):
    return len(self.parent) == 0

  def __repr__(self):
    return self.word


def preorder_traversal(root):
  nodes = [root]
  for node in root.child:
    nodes += preorder_traversal(node)
  return nodes


def tok_to_tree(toks, root):
  nodes = [root]
  parent = root
  for i in range(len(toks)):
    if '[' in toks[i]:
      node = Node(toks[i], parent, index=len(nodes))
      nodes.append(node)
      parent.child.append(node)
      parent = node
    elif ']' in toks[i]:
      parent = parent.parent[0]
    else:
      node = Node(toks[i], parent, index=len(nodes))
      nodes.append(node)
      parent.child.append(node)
  return nodes


def tree_to_data(nodes, multiple=False):
  words = []
  if not multiple:
    edges = []
  else:
    edges = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
  for node in nodes:
    if node.is_root():
      words.append(node.word)
      if not multiple:
        edges.append(-1)
    else:
      words.append(node.word)
      if not multiple:
        edges.append(node.parent[0].index)
      else:
        for p in node.parent:
          edges[node.index][p.index] = 1
  return words, edges


def is_valid_tree(pred, src=None, pred_id=None, tgt_vocab=None):

  num_left_brackets = ' '.join(pred).count('[')
  num_right_brackets = ' '.join(pred).count(']')
  if num_left_brackets != num_right_brackets:
    return False

  a = 0
  for c in ' '.join(pred):
    if c == '[':
      a += 1
    elif c == ']':
      a -= 1
    if a < 0:
      return False
  if a != 0:
    return False

  if src is not None:
    assert pred_id is not None
    assert tgt_vocab is not None
    src_len = sum(tok != '<PAD>' for tok in src)
    src_id = np.arange(len(src[:src_len]))

    pred_src_id = [(i - len(tgt_vocab)) for i in pred_id if i >= len(tgt_vocab)]
    pred_src_id.sort()
    if len(pred_src_id) != src_len:
      return False

    for i in range(len(pred_src_id)):
      if src_id[i] != pred_src_id[i]:
        return False

  return True


def tree_to_map(tree):
  return {node.identifier: node for node in tree}


def tree_to_map_multiple(tree):
  ans = {}
  for node in tree:
    if node.identifier in ans:
      ans[node.identifier].append(node)
    else:
      ans[node.identifier] = [node]
  return ans


def merge_two_trees(tree1, root2):
  tree1_map = tree_to_map(tree1)
  if not root2.is_root():
    parent1 = tree1_map[root2.parent[0].identifier]
    add = True
    for child in parent1.child:
      if child.identifier == root2.identifier:
        add = False
    if add:
      if root2.identifier in tree1_map:
        self1 = tree1_map[root2.identifier]
        parent1.child.append(self1)
        self1.parent.append(parent1)
      else:
        new_node = Node(root2.word,
                        parent1,
                        index=len(tree1),
                        signature=root2.signature,
                        identifier=root2.identifier)
        tree1.append(new_node)
        parent1.child.append(new_node)
  for child in root2.child:
    merge_two_trees(tree1, child)


def merge_tree(trees, use_signature=True):
  orig = trees[0]
  if use_signature:
    for tree in trees:
      set_signature(tree[0])
  for i in range(1, len(trees)):
    tree = trees[i]
    merge_two_trees(orig, tree[0])
  return orig


def set_signature(node):
  if not node.is_leaf():
    sig = []
    for child in node.child:
      set_signature(child)
      sig += child.signature
    node.signature = set([int(i) for i in sig])
    node.identifier = node.word + '|' + ' '.join(
        [str(i) for i in list(node.signature)])
  else:
    if not node.word.isnumeric():
      node.signature = []


def set_signature_graph(node):
  node.visited = True
  if not node.is_leaf():
    sig = []
    for child in node.child:
      if not child.visited:
        set_signature_graph(child)
      if child.sig_set:
        sig += child.signature
    node.signature = set([int(i) for i in sig])
    node.identifier = node.word + '|' + ' '.join(
        [str(i) for i in list(node.signature)])
    node.sig_set = True
  else:
    node.sig_set = True
    if not node.word.isnumeric():
      node.signature = []


def get_boundary(node):
  if node.is_leaf():
    left = int(node.signature[0])
    right = int(node.signature[0])
  else:
    left = list(node.signature)[0]
    right = list(node.signature)[-1]
  return left, right


def node_sort(child):
  new_child = []
  for i in range(len(child)):
    node1 = child[i]
    left1, right1 = get_boundary(node1)
    pos = 0
    for j in range(len(new_child)):
      node2 = new_child[j]
      left2, right2 = get_boundary(node2)
      if right1 > right2:
        pos += 1
      elif right1 == right2:
        if left1 > left2:
          pos += 1
        elif left1 == left2:
          pos += 1
        else:
          break
      else:
        break
    new_child.insert(pos, node1)
  return new_child


def topological_sort(root, sort_node=False):
  cycle = False
  results = []
  root.visited = True
  if sort_node:
    root.child = node_sort(root.child)
  for i in range(len(root.child) - 1, -1, -1):
    child = root.child[i]
    if not child.visited:
      result, has_cycle = topological_sort(child, sort_node)
      results += result
      cycle = cycle or has_cycle
    elif not child.sig_set:
      cycle = True

  root.sig_set = True
  results.append(root)
  return results, cycle


def set_index(tree):
  for i in range(len(tree)):
    tree[i].index = i


def reconstruct_tree(tokens, edges, length, vocab, with_bos=False):
  nodes = []
  if with_bos:
    nodes.append(Node(vocab.BOS, None))

  for i in range(length):
    if tokens[i] < len(vocab):
      curr = Node(vocab.idx2token[tokens[i]], None, index=len(nodes))
    else:
      curr = Node(str(tokens[i] - len(vocab)), None, index=len(nodes))
    nodes.append(curr)
    if i != 0 or with_bos:
      for j in range(len(edges[i][:length])):
        if edges[i][j] == 1:
          nodes[j].child.append(curr)
          curr.parent.append(nodes[j])
  set_signature_graph(nodes[0])
  for n in nodes:
    n.visited = False
    n.sig_set = False
  _, has_cycle = topological_sort(nodes[0])
  return nodes, has_cycle


def contain_tree(root1, root2):
  ans = True
  map = tree_to_map(root1.child)
  for c in root2.child:
    if c.identifier in map:
      ans = (ans and contain_tree(map[c.identifier], c))
    else:
      return False
  return ans


def sort_tree(tree):
  new_tree = []
  source_nodes = []
  for node in tree:
    if not node.word.isnumeric():
      new_tree.append(node)
    else:
      source_nodes.append(node)
  source_nodes = sorted(source_nodes, key=lambda node: int(node.word))
  new_tree += source_nodes
  return new_tree
