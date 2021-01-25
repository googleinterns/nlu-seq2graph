"""Copyright 2020 Google LLC Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""Utils for graph operations."""

from collections import defaultdict
from copy import deepcopy
from itertools import product

from absl import logging
import numpy as np


class Node(object):

  def __init__(self, word):
    self.word = word
    self.parent = []
    self.child = []
    # signautre is a tuple of (word, left_boundary, right_boundary).
    self.signature = None
    # used internally
    self.valid_child_combinations = None
    self.derivs = None

  def is_leaf(self):
    return len(self.child) == 0

  def is_root(self):
    return len(self.parent) == 0

  def __repr__(self):
    return self.word


class SignatureGraph(object):

  def __init__(self, root_signature):
    self.root_signature = root_signature
    # nodes is a map of node_signature -> (set of parent node signatures,
    # set of child node signatures).
    self.nodes = {}

  def try_adding_tree(self, tree, check_tree=True, check_connectness=True):
    new_nodes = deepcopy(self.nodes)
    for node in tree:
      if node.signature not in new_nodes:
        new_nodes[node.signature] = (set(), set())
      parents, children = new_nodes[node.signature]
      if node.parent:
        if check_tree:
          assert len(node.parent) == 1
        for p in node.parent:
          if p.signature != node.signature:
            # Skip duplicated nodes from prediction.
            parents.add(p.signature)
      for c in node.child:
        if c.signature != node.signature:
          children.add(c.signature)

    if not self.is_valid_dag(new_nodes, check_connectness):
      return False
    self.nodes = new_nodes
    return True

  def contains_tree(self, tree, verbose=False):
    calc_signature(tree[0])
    for node in tree:
      if node.signature not in self.nodes:
        if verbose:
          print(node.signature, "not found")
        return False
      for p in node.parent:
        if p.signature not in self.nodes[node.signature][0]:
          if verbose:
            print(node.signature, "parent", p.signature, "not found")
          return False
      for c in node.child:
        if c.signature not in self.nodes[node.signature][1]:
          if verbose:
            print(node.signature, "child", c.signature, "not found")
          return False
    return True

  def is_valid_dag(self, nodes, check_connectness=True):
    # check if all parent and child signatures are in the graph
    for parents, children in nodes.values():
      if (any(p not in nodes for p in parents) or
          any(c not in nodes for c in children)):
        print('mismatch parents/children')
        return False

    visited = set()

    def contains_loop(node_signature, path):
      visited.add(node_signature)
      if node_signature in path:
        print('loop detected', node_signature, path)
        return True
      new_path = path + [node_signature]
      for c in nodes[node_signature][1]:
        if contains_loop(c, new_path):
          return True
      return False

    if contains_loop(self.root_signature, []):
      print("loop detected", nodes)
      return False

    if check_connectness and len(visited) != len(nodes):
      print('visited', len(visited), '# nodes', len(nodes))
      return False

    return True

  def topo_sort2(self):
    """DFS based topo sort."""
    # Add dummy dependency between leaf nodes to guarantee their order.
    new_nodes = deepcopy(self.nodes)
    for n, (parents, _) in new_nodes.items():
      if n[2].isnumeric() and n[0] != 0:
        left_leaf = (n[0] - 1, n[1] - 1, str(int(n[2]) - 1))
        new_nodes[left_leaf][1].add(n)
        parents.add(left_leaf)

    # Sort the children for each node.
    sorted_children = dict(
        (n, sorted(list(c))) for n, (_, c) in new_nodes.items())

    sorted_nodes = []

    def _topo_sort(node):
      if node in sorted_nodes:
        return
      parents = new_nodes[node][0]
      if any(p not in sorted_nodes for p in parents):
        return
      # all parents are visited
      sorted_nodes.append(node)
      for c in sorted_children[node]:
        _topo_sort(c)

      return

    _topo_sort(self.root_signature)

    return sorted_nodes

  def topo_sort(self):
    # Add dummy dependency between leaf nodes to guarantee their order.
    new_nodes = deepcopy(self.nodes)
    for n, (parents, _) in new_nodes.items():
      if n[2].isnumeric() and n[0] != 0:
        left_leaf = (n[0] - 1, n[1] - 1, str(int(n[2]) - 1))
        new_nodes[left_leaf][1].add(n)
        parents.add(left_leaf)

    # Sort the children for each node.
    sorted_children = dict(
        (n, sorted(list(c))) for n, (_, c) in new_nodes.items())

    visited = set()

    def _topo_sort(node):
      sorted_nodes = []
      for c in reversed(sorted_children[node]):
        if c not in visited:
          sorted_c = _topo_sort(c)
          sorted_nodes += sorted_c
      visited.add(node)
      sorted_nodes.append(node)

      return sorted_nodes

    sorted_nodes = list(reversed(_topo_sort(self.root_signature)))

    assert (len(sorted_nodes) == len(self.nodes))

    return sorted_nodes


def tok_to_tree(tokens, root):
  """Build tree from a sequence of tokens with the given root."""
  nodes = [root]
  parent = root
  for tok in tokens:
    if '[' in tok:
      # internal nodes
      node = Node(tok)
      nodes.append(node)
      node.parent.append(parent)
      parent.child.append(node)
      parent = node
    elif ']' in tok:
      parent = parent.parent[0]
    else:
      node = Node(tok)
      nodes.append(node)
      node.parent.append(parent)
      parent.child.append(node)
  return nodes


def calc_signature(node):
  """Calculate the signature for each tree node."""
  if node.signature:
    return node.signature
  if not node.is_leaf():
    for child in node.child:
      calc_signature(child)
    left_bound = min([child.signature[0] for child in node.child])
    right_bound = max([child.signature[1] for child in node.child])
    node.signature = (left_bound, right_bound, node.word)
  elif node.word.isnumeric():
    node.signature = (int(node.word), int(node.word) + 1, node.word)
  else:
    logging.error("Incorrect leaf node {}".format(node.word))


def is_valid_sequence(pred, src=None, pred_id=None, tgt_vocab=None):

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


def is_valid_tree(tree_nodes):
  last_src_idx = -1
  for node in tree_nodes:
    if node.is_leaf():
      if node.word.isnumeric() and int(node.word) == last_src_idx + 1:
        last_src_idx += 1
      else:
        return False
  return True


def graph_to_tfex(sorted_nodes, sig_graph, tgt_vocab, append_eos=True):
  words = [n[2] for n in sorted_nodes] + [tgt_vocab.EOS]
  edges = np.zeros((len(words), len(words)), dtype=np.int64)
  for nid, node in enumerate(sorted_nodes):
    parents = sig_graph.nodes[node][0]
    for p in parents:
      pid = sorted_nodes.index(p)
      assert pid < nid
      edges[(nid, pid)] = 1
  edges[-1][0] = 1  # EOS points to BOS
  return words, edges


def reconstruct_tree(tokens, edges, length, vocab, robust_mode=False):
  """Rebuild tree from model predictions.

  Note that the beginning BOS in tokens and edges are not included. So if the
  length of tokens is L, the shape of edges will be Lx(L+1).

  If robust_mode is True, some heuristics will be applied to remove invalid
  nodes. Otherwise, it will check if the tree is valid.
  """

  def remove_invalid_leaves(nodes):
    invalid_nodes = []
    for n in reversed(nodes):
      if n.is_leaf() and not n.word.isnumeric():
        invalid_nodes.append(n)
        for p in n.parent:
          p.child.remove(n)
    return [n for n in nodes if n not in invalid_nodes]

  covered_leaves = {}

  def calc_and_fix_signature(root):
    """Returns the leaves covered by the node. Removes a child if the child leads to a discontinuous span."""
    if root in covered_leaves:
      return covered_leaves[root]
    if root.is_leaf():
      root.signature = (int(root.word), int(root.word) + 1, root.word)
      covered_leaves[root] = set([int(root.word)])
      return covered_leaves[root]
    leaves = set()
    for c in root.child:
      leaves |= calc_and_fix_signature(c)
    # find the true boundary that covers a continuous span
    left = min(leaves)
    right = left
    while right in leaves:
      right += 1
    child = [
        c for c in root.child
        if c.signature[0] >= left and c.signature[1] <= right
    ]
    root.child = child
    root.signature = (left, right, root.word)
    covered_leaves[root] = set(range(left, right))
    return covered_leaves[root]

  root = Node(vocab.BOS)

  nodes = [root]

  for i in range(length):
    if tokens[i] == vocab.token2idx[vocab.EOS]:
      break
    if tokens[i] < len(vocab):
      curr = Node(vocab.idx2token[tokens[i]])
    else:
      curr = Node(str(tokens[i] - len(vocab)))
    nodes.append(curr)
    for j in range(i + 1):
      if edges[i][j] == 1:
        nodes[j].child.append(curr)
        curr.parent.append(nodes[j])

  if robust_mode:
    nodes = remove_invalid_leaves(nodes)
  elif not is_valid_tree(nodes):
    return []
  if nodes:
    calc_and_fix_signature(nodes[0])
    # calc_signature(nodes[0])
  return nodes


def contains_tree(root1, root2):
  """Check if root1 contains root2."""
  ans = True
  root1_map = {node.signature: node for node in root1.child}
  for c in root2.child:
    if c.signature in root1_map:
      ans = (ans and contains_tree(root1_map[c.signature], c))
    else:
      return False
  return ans


def precompute_children_combinations(root):
  if root.is_leaf():
    return True
  if root.valid_child_combinations is not None:
    return root.valid_child_combinations
  child = []
  for c in root.child:
    if c.is_leaf():
      child.append(c)
    elif precompute_children_combinations(c):
      child.append(c)

  lefts = defaultdict(list)
  for c in child:
    lefts[c.signature[0]].append(c)
  # find all combinations of children that fills the whole span of root.
  # list of candiadte. each candidate is a list of (left, right, child)
  combinations = []
  candidates = [[(0, root.signature[0], None)]]
  while candidates:
    new_candidates = []
    for c in candidates:
      right = c[-1][1]
      for n in lefts[right]:
        if n.signature[1] == root.signature[1]:  # finished
          combinations.append(c + [(n.signature[0], n.signature[1], n)])
        else:
          new_candidates.append(c + [(n.signature[0], n.signature[1], n)])
    candidates = new_candidates
  root.valid_child_combinations = combinations
  return root.valid_child_combinations


def retrieve_trees(root, max_num_derivs=10):
  if root.is_leaf():
    yield root.word
    return
  if root.derivs is None:
    derivs = []
    finished = False
    for combination in root.valid_child_combinations:
      iters = [
          retrieve_trees(c, max_num_derivs) for (_, _, c) in combination[1:]
      ]
      for eid, element in enumerate(product(*iters)):
        if len(derivs) == max_num_derivs:
          finished = True
          break
        ret = [root.word]
        for e in element:
          if type(e) is list:
            ret += e
          else:
            ret.append(e)
        if root.word.startswith('['):
          ret.append(']')
        derivs.append(ret)
      if finished:
        break
    root.derivs = derivs
  for d in root.derivs:
    yield d


if __name__ == '__main__':
  g = SignatureGraph((0, 3, 'ROOT'))
  g.nodes = {
      (0, 3, 'ROOT'): (set(), {(0, 2, 'A'), (2, 3, 'B'), (0, 3, 'C')}),
      (0, 2, 'A'): ({(0, 3, 'ROOT')}, {(0, 1, '0'), (1, 2, '1')}),
      (2, 3, 'B'): ({(0, 3, 'ROOT')}, {(2, 3, '2')}),
      (0, 3, 'C'): ({(0, 3, 'ROOT')}, {(0, 1, '0'), (1, 2, '1'), (2, 3, '2')}),
      (0, 1, '0'): ({(0, 2, 'A'), (0, 3, 'C')}, set()),
      (1, 2, '1'): ({(0, 2, 'A'), (0, 3, 'C')}, set()),
      (2, 3, '2'): ({(2, 3, 'B'), (0, 3, 'C')}, set())
  }
  print(g.topo_sort())
