from itertools import combinations
import sys,json

class Markable:
  def __init__(self, doc_name, start, end, MIN, is_referring, words, is_split_antecedent=False, sub_markables=None):
    self.doc_name = doc_name
    self.start = start
    self.end = end
    self.MIN = MIN
    self.is_referring = is_referring
    self.words = words
    self.is_split_antecedent = is_split_antecedent
    self.sub_markables = sub_markables

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      # MIN is only set for the key markables
      if self.MIN:
        return (self.doc_name == other.doc_name
                and other.start[0] >= self.start[0]
                and other.start[0] <= self.MIN[0]
                and other.end[-1] <= self.end[-1]
                and other.end[-1] >= self.MIN[1])
      elif other.MIN:
        return (self.doc_name == other.doc_name
                and self.start[0] >= other.start[0]
                and self.start[0] <= other.MIN[0]
                and self.end[-1] <= other.end[-1]
                and self.end[-1] >= other.MIN[1])
      else:
        return (self.doc_name == other.doc_name
                and self.start == other.start
                and self.end == other.end)
    return NotImplemented

  def __neq__(self, other):
    if isinstance(other, self.__class__):
      return self.__eq__(other)
    return NotImplemented

  def __hash__(self):
    return hash(frozenset((int(''.join([str(s) for s in self.start])), int(''.join([str(s) for s in self.end])))))

  def __str__(self):
    return ('DOC: {} SPAN: ({}, {}) String: {} MIN: {}, Referring tag: {} Plural tag: {}'.format(
      self.doc_name, self.start, self.end, [' '.join(words) for words in self.words], self.MIN,
      self.is_referring, 'split_antecedent' if self.is_split_antecedent else 'singular'))

class Evaluator:
  def __init__(self, beta=1, split_antecedent_importance=1, keep_aggregated_values=False):
    self.p_num = 0
    self.p_den = 0
    self.r_num = 0
    self.r_den = 0
    self.beta = beta
    self.split_antecedent_importance = split_antecedent_importance

    if keep_aggregated_values:
      self.aggregated_p_num = []
      self.aggregated_p_den = []
      self.aggregated_r_num = []
      self.aggregated_r_den = []

  def update(self, key_doc, sys_doc):
    key_clusters, sys_clusters, key_mention_sys_cluster,sys_mention_key_cluster = get_coref_info_from_doc(key_doc, sys_doc)

    pn, pd = lea(sys_clusters, key_clusters,
                 sys_mention_key_cluster, self.split_antecedent_importance)
    rn, rd = lea(key_clusters, sys_clusters,
                 key_mention_sys_cluster, self.split_antecedent_importance)
    self.p_num += pn
    self.p_den += pd
    self.r_num += rn
    self.r_den += rd

  def get_f1(self):
    return f1(self.p_num,
              self.p_den,
              self.r_num,
              self.r_den,
              beta=self.beta)

  def get_recall(self):
    return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

  def get_precision(self):
    return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

  def get_prf(self):
    return self.get_precision(), self.get_recall(), self.get_f1()

  def get_counts(self):
    return self.p_num, self.p_den, self.r_num, self.r_den

  def get_aggregated_values(self):
    return (self.aggregated_p_num, self.aggregated_p_den,
            self.aggregated_r_num, self.aggregated_r_den)

def get_coref_info_from_doc(key_doc, sys_doc):
  doc_name = key_doc['doc_key']
  assert key_doc['doc_key'] == sys_doc['doc_key']
  all_words = [t for sent in key_doc['sentences'] for t in sent]
  key_clusters = {}
  key_mention2cluster = {}
  for cid, cl in enumerate(key_doc['clusters']):
    if len(cl) == 1 and cl[0][2] == 1:
      continue #exclude non_referring
    key_clusters[cid] = []
    for mention in cl:
      m_span = (mention[0],mention[1])
      key_mention2cluster[m_span] = cid
      m = Markable(
        doc_name, [mention[0]],
        [mention[1]], None,
        'referring',
        [all_words[mention[0]:
                   mention[1] + 1]],
        False)
      key_clusters[cid].append(m)
  key_split_antecedent_map = {}
  for anaphora, antecedent in key_doc['split_antecedents']:
    anaphora = tuple(anaphora)
    antecedent = tuple(antecedent)
    if not anaphora in key_split_antecedent_map:
      key_split_antecedent_map[anaphora] = []
    key_split_antecedent_map[anaphora].append(antecedent)
  for anaphora in key_split_antecedent_map:
    antecedent = sorted(key_split_antecedent_map[anaphora])
    antecedent_starts = [start for start,_ in antecedent]
    antecedent_ends = [end for _,end in antecedent]
    cid = key_mention2cluster[anaphora]
    m = Markable(doc_name, antecedent_starts, antecedent_ends, None, 'referring',
                          [all_words[s:e + 1] for s, e in zip(antecedent_starts, antecedent_ends)], True)
    m.sub_markables = []
    subset_ids = get_sub_lists(len(m.start))
    for subset in subset_ids:
      start = []
      end = []
      words = []
      for id in subset:
        start.append(m.start[id])
        end.append(m.end[id])
        words.append(m.words[id])
      sub_m = Markable(
        m.doc_name, start, end, None,
        m.is_referring, words, True if len(subset) > 1 else False)
      m.sub_markables.append(sub_m)
    m.sub_markables.sort(key=lambda x: len(x.start), reverse=True)
    key_clusters[cid].append(m)

  sys_clusters = {}
  sys_mention2cluster = {}
  for cid, cl in enumerate(sys_doc['clusters']):
    if len(cl) == 1 and cl[0][2] == 1:
      continue #exclude non_referring
    sys_clusters[cid] = []
    for mention in cl:
      m_span = (mention[0], mention[1])
      sys_mention2cluster[m_span] = cid
      m = Markable(
        doc_name, [mention[0]],
        [mention[1]], None,
        'referring',
        [all_words[mention[0]:
                   mention[1] + 1]],
        False)
      sys_clusters[cid].append(m)

  sys_split_antecedent_map = {}
  for anaphora, antecedent in sys_doc['split_antecedents']:
    anaphora = tuple(anaphora)
    antecedent = tuple(antecedent)
    if not anaphora in sys_split_antecedent_map:
      sys_split_antecedent_map[anaphora] = []
    sys_split_antecedent_map[anaphora].append(antecedent)
  for anaphora in sys_split_antecedent_map:
    antecedent = sorted(sys_split_antecedent_map[anaphora])
    antecedent_starts = [start for start,_ in antecedent]
    antecedent_ends = [end for _, end in antecedent]
    cid = sys_mention2cluster[anaphora]
    m = Markable(doc_name, antecedent_starts, antecedent_ends, None, 'referring',
                          [all_words[s:e + 1] for s, e in zip(antecedent_starts, antecedent_ends)], True)
    m.sub_markables = []
    subset_ids = get_sub_lists(len(m.start))
    for subset in subset_ids:
      start = []
      end = []
      words = []
      for id in subset:
        start.append(m.start[id])
        end.append(m.end[id])
        words.append(m.words[id])
      sub_m = Markable(
        m.doc_name, start, end, None,
        m.is_referring, words, True if len(subset) > 1 else False)
      m.sub_markables.append(sub_m)
    m.sub_markables.sort(key=lambda x: len(x.start), reverse=True)
    sys_clusters[cid].append(m)

  key_clusters = [key_clusters[cid] for cid in key_clusters]
  sys_clusters = [sys_clusters[cid] for cid in sys_clusters]

  sys_mention_key_cluster = get_markable_assignments(
    sys_clusters, key_clusters)
  key_mention_sys_cluster = get_markable_assignments(
    key_clusters, sys_clusters)

  return key_clusters, sys_clusters, key_mention_sys_cluster, sys_mention_key_cluster

def get_markable_assignments(inp_clusters, out_clusters):
  markable_cluster_ids = {}
  out_dic = {}
  all_out_markables = []

  for cluster in out_clusters:
    for m in cluster:
      if m not in all_out_markables:
        all_out_markables.append(m)

  for cluster_id, cluster in enumerate(out_clusters):
    for m in cluster:
      out_dic[m] = (cluster_id, 1)
      if m.is_split_antecedent:
        for sub_m in m.sub_markables:
          if sub_m not in all_out_markables:
            out_dic[sub_m] = (cluster_id, len(sub_m.start) / float(len(m.start)))

  for cluster in inp_clusters:
    for im in cluster:
      if im in out_dic:
        if im not in markable_cluster_ids:
          markable_cluster_ids[im] = [out_dic[im]]
        elif out_dic[im] not in markable_cluster_ids[im]:
          if out_dic[im][1] == 1:
            markable_cluster_ids[im].insert(0, out_dic[im])
          else:
            markable_cluster_ids[im].append(out_dic[im])
      elif im.is_split_antecedent:
        for s in im.sub_markables:
          if s in out_dic:
            oratio = out_dic[s][1]
            cratio = len(s.start) / float(len(im.start))
            cluster, ratio = out_dic[s][0], oratio * cratio
            if s not in markable_cluster_ids:
              markable_cluster_ids[s] = [(cluster, ratio)]
            elif (cluster, ratio) not in markable_cluster_ids[s]:
              markable_cluster_ids[s].append((cluster, ratio))
            break

  for m in markable_cluster_ids:
    markable_cluster_ids[m].sort(key=lambda x: x[1], reverse=True)

  return markable_cluster_ids

def get_sub_lists(idx):
    idx_list = [i for i in range(idx)]
    subs = []
    for i in range(1, len(idx_list)):
        temp = [list(x) for x in combinations(idx_list, i)]
        if len(temp)>0:
            subs.extend(temp)
    return subs

def f1(p_num, p_den, r_num, r_den, beta=1):
  p = 0 if p_den == 0 else p_num / float(p_den)
  r = 0 if r_den == 0 else r_num / float(r_den)
  return (0 if p + r == 0
          else (1 + beta * beta) * p * r / (beta * beta * p + r))

def compute_common_links(c, mention_to_gold):
  common_links = 0
  for i, m in enumerate(c):
    m_cluster_ratio = get_cluster_ratio(m, mention_to_gold)
    if m_cluster_ratio:
      for m2 in c[i + 1:]:
        m2_cluster_ratio = get_cluster_ratio(m2, mention_to_gold)
        ratio = same_cluster(m_cluster_ratio, m2_cluster_ratio)
        if ratio:
          common_links += ratio
  return common_links


def get_cluster_ratio(m, mention_to_gold):
  if m in mention_to_gold:
    return mention_to_gold[m]
  if m.is_split_antecedent:
    for s in m.sub_markables:
      if s in mention_to_gold:
        return mention_to_gold[s]
  return None


def same_cluster(cluster_ratio1, cluster_ratio2):
  if cluster_ratio1 and cluster_ratio2:
    for c1, r1 in cluster_ratio1:
      for c2, r2 in cluster_ratio2:
        if c1 == c2:
          return r1 * r2
  return None


def has_split_antecedent(c):
  for m in c:
    if m.is_split_antecedent:
      return True
  return False


def lea(input_clusters, output_clusters, mention_to_gold, split_antecedent_importance=1):
  num, den = 0, 0
  for c in input_clusters:
    if len(c) == 1:
      all_links = 1
      if c[0] in mention_to_gold and len(
          output_clusters[mention_to_gold[c[0]][0][0]]) == 1:
        common_links = 1
      else:
        common_links = 0
    else:
      all_links = len(c) * (len(c) - 1) / 2.0
      common_links = compute_common_links(c, mention_to_gold)

    is_split_antecedent = has_split_antecedent(c)
    num += (split_antecedent_importance if is_split_antecedent else 1) * len(c) * common_links / float(all_links)
    den += (split_antecedent_importance if is_split_antecedent else 1) * len(c)
  return num, den

def main():
  key_json = sys.argv[1]
  sys_json = sys.argv[2]
  split_antecedent_importance = int(sys.argv[3]) if len(sys.argv) > 3 else 1

  key_docs = [json.loads(line) for line in open(key_json)]
  sys_docs = [json.loads(line) for line in open(sys_json)]
  lea_evaluator = Evaluator(split_antecedent_importance=split_antecedent_importance)
  for key_doc,sys_doc in zip(key_docs,sys_docs):
    if len(key_doc['split_antecedents']) > 0:
      lea_evaluator.update(key_doc,sys_doc)
  precision, recall, f1 = lea_evaluator.get_prf()
  print('Recall: %.2f' % (recall * 100), ' Precision: %.2f' % (precision * 100), ' F1: %.2f' % (f1 * 100))
  print '%.1f\t%.1f\t%.1f' % (recall * 100, precision * 100, f1 * 100)

main()