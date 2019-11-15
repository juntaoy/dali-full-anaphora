from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
import coref_ops
import conll,crac
import metrics

class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.max_span_width = config["max_span_width"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None # Load eval data lazily.
    self.n_types = config['n_types']
    self.max_mention_per_cluster = 20
    #non Mention, NRs, DN
    #(only NRs can be extended, non mention always be 0  DN must be the last one)
    self.crac_doc = self.config["crac_doc"]

    input_props = []
    input_props.append((tf.string, [None, None])) # Tokens.
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
    input_props.append((tf.int32, [None, None, None])) # Character indices.
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.int32, [None])) # Speaker IDs.
    input_props.append((tf.int32, [])) # Genre.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None])) # Cluster ids.
    input_props.append((tf.int32, [None]))  # Gold types

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
    with open(self.config["train_path"]) as f:
      train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    def _enqueue_loop():
      while True:
        random.shuffle(train_examples)
        for example in train_examples:
          tensorized_example = self.tensorize_example(example, is_training=True)
          feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
          session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session,model_name="model.max.ckpt"):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], model_name)
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_mentions(self, mentions):
    starts, ends, types = [], [], []
    for m in mentions:
      starts.append(m[0])
      ends.append(m[1])
      types.append(m[2] if self.crac_doc else -1)

    return np.array(starts), np.array(ends), np.array(types)


  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
      for mention in cluster:
        cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = util.flatten(example["speakers"])

    assert num_words == len(speakers)

    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    tokens = np.array(tokens)

    speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
    speaker_ids = np.array([speaker_dict[s] for s in speakers])

    doc_key = example["doc_key"]
    genre = self.genres[doc_key[:2]]
    gold_starts, gold_ends,gold_types = self.tensorize_mentions(gold_mentions)

    lm_emb = self.load_lm_embeddings(doc_key)

    example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids,gold_types)

    if is_training and num_words > self.config["max_training_words"]:
      return self.truncate_example(*example_tensors)
    else:
      return example_tensors

  def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids,gold_types):

    num_sentences = context_word_emb.shape[0]
    max_training_sentences = num_sentences

    num_words = sum(text_len)
    assert num_words > self.config["max_training_words"]


    while num_words > self.config["max_training_words"]:
      max_training_sentences -= 1
      sentence_offset = random.randint(0, num_sentences - max_training_sentences)
      word_offset = text_len[:sentence_offset].sum()
      num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()

    tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
    context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
    head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
    lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
    char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    speaker_ids = speaker_ids[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]
    gold_types = gold_types[gold_spans]

    return tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids,gold_types


  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)


  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]

  def get_mention_scores(self, span_emb, out_num=1,scope_name="mention_scores"):
    with tf.variable_scope(scope_name):
      return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], out_num, self.dropout) # [k, out_num]

  def softmax_loss(self, cluster_scores, gold_labels):
    marginalized_gold_scores = tf.reduce_logsumexp(cluster_scores + tf.log(tf.to_float(gold_labels)), [1])
    log_norm = tf.reduce_logsumexp(cluster_scores, [1])
    loss = log_norm - marginalized_gold_scores
    return loss


  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)


  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                          tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
    same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                        tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
    same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
    candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
    candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
    return candidate_labels


  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in range(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return self.flatten_emb_by_sentence(text_outputs, text_len_mask)


  def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, gold_cluster_ids, gold_types):
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if not self.lm_file:
      raise LookupError("BERT embeddings need to be computed before hand!")

    lm_emb_size = util.shape(lm_emb, 2)
    lm_num_layers = util.shape(lm_emb, 3)
    with tf.variable_scope("lm_aggregation"):
      self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
      self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
    flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
    flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
    aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
    aggregated_lm_emb *= self.lm_scaling
    context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask) # [num_words, emb]
    num_words = util.shape(context_outputs, 0)

    genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]

    sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
    flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

    candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) # [num_words, max_span_width]
    candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) # [num_words, max_span_width]
    candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) # [num_words, max_span_width]
    candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) # [num_words, max_span_width]
    candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) # [num_words, max_span_width]
    flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
    candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask) # [num_candidates]
    candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask) # [num_candidates]

    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                      gold_cluster_ids)  # [num_candidates]

    candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]
    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb) # [k, 1]
    candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [k]

    k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
    top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                               tf.expand_dims(candidate_starts, 0),
                                               tf.expand_dims(candidate_ends, 0),
                                               tf.expand_dims(k, 0),
                                               util.shape(context_outputs, 0),
                                               True) # [1, k]
    top_span_indices.set_shape([1, None])
    top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

    top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
    top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
    top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
    top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
    top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
    top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]

    mention_type_scores = self.get_mention_scores(top_span_emb, self.n_types,"other_socring")  # [k, n_types]
    expanded_top_span_mention_scores = tf.expand_dims(top_span_mention_scores,axis=1)
    dim_top_span_mention_scores = tf.concat([tf.zeros_like(expanded_top_span_mention_scores,dtype=tf.float32),tf.tile(expanded_top_span_mention_scores,[1,self.n_types-1])],axis=1) #[k,n_types]
    mention_type_scores += dim_top_span_mention_scores



    with tf.variable_scope("mention_attention"):
      if self.config["use_cluster_position_emb"]:
        cluster_position_emb = tf.gather(tf.get_variable("cluster_position_emb",[self.max_mention_per_cluster,self.config["feature_size"]]),tf.range(0,self.max_mention_per_cluster))
        tiled_cluster_position_emb = tf.tile(tf.expand_dims(cluster_position_emb,0),[k,1,1])
        tiled_top_span_emb = tf.tile(tf.expand_dims(top_span_emb,1),[1,self.max_mention_per_cluster,1])
        top_span_emb_with_cluster_position = tf.concat([tiled_top_span_emb,tiled_cluster_position_emb],2)
        top_span_att_scores = util.projection(top_span_emb_with_cluster_position,1)
        top_span_att_scores = tf.squeeze(top_span_att_scores,2) #[k,max_mention_per_cluster]
      else:
        top_span_att_scores = util.projection(top_span_emb, 1)
        top_span_att_scores = tf.squeeze(top_span_att_scores, axis=1)#[num_mention]

    if self.config["train_on_oracle_cluster"]:
      cluster_scores, cluster_indices,individual_cluster_size,predicted_antecedents, predicted_mention_types = \
        self.get_oracle_cluster_scores(top_span_starts, top_span_ends, top_span_cluster_ids, top_span_emb,
                                     top_span_mention_scores, top_span_att_scores, top_span_speaker_ids,
                                     genre_emb, mention_type_scores,is_training)
    else:
      cluster_scores, cluster_indices, individual_cluster_size, predicted_antecedents, predicted_mention_types = \
        self.get_cluster_scores(top_span_emb, top_span_mention_scores, top_span_att_scores, top_span_speaker_ids,
                              genre_emb, mention_type_scores,is_training)


    gold_labels = coref_ops.gold_scores(
      mention_starts=top_span_starts,
      mention_ends=top_span_ends,
      mention_type_scores=mention_type_scores,
      gold_starts=gold_starts,
      gold_ends=gold_ends,
      gold_cluster_ids=gold_cluster_ids,
      gold_types=gold_types,
      crac_doc = self.crac_doc,
      cluster_indices=cluster_indices,
      cluster_size=individual_cluster_size,
      n_types=self.n_types
    )
    gold_labels.set_shape([None, None])

    loss = tf.reduce_sum(self.softmax_loss(cluster_scores, gold_labels))

    return [top_span_starts, top_span_ends, predicted_antecedents,predicted_mention_types], loss


  def get_oracle_cluster_scores(self, mention_starts,mention_ends,mention_cluster_ids, mention_emb, mention_scores,mention_att_scores, mention_speaker_ids, genre_emb,mention_type_scores,is_training):
    num_mentions = util.shape(mention_scores, 0)
    max_clusters = tf.minimum(num_mentions, self.config["max_top_antecedents"])


    oracle_clusters, oracle_cluster_size = coref_ops.oracle_clusters(
      mention_starts=mention_starts,
      mention_ends=mention_ends,
      mention_cluster_ids=mention_cluster_ids,
      max_cluster_size=self.max_mention_per_cluster
    )
    oracle_clusters.set_shape([None, self.max_mention_per_cluster])
    oracle_cluster_size.set_shape([None])

    oracle_cluster_width_bin = coref_ops.cluster_width_bins(widths=oracle_cluster_size)
    oracle_cluster_width_bin.set_shape([None])


    if self.config["use_cluster_position_emb"]:
      oracle_clusters_with_position = tf.concat([tf.expand_dims(oracle_clusters, 2), tf.tile(
        tf.expand_dims(tf.expand_dims(tf.range(0, self.max_mention_per_cluster), 0), 2), [num_mentions, 1, 1])],
                                                axis=2)  # [k,max_mention_per_cluster,2]
      oracle_cluster_att_scores = tf.gather_nd(mention_att_scores,
                                               oracle_clusters_with_position)  # [k,max_mention_per_cluster]
    else:
      oracle_cluster_att_scores = tf.gather(mention_att_scores, oracle_clusters)  # [k,max_mention_per_cluster]

    oracle_cluster_att_scores += tf.log(tf.sequence_mask(oracle_cluster_size, self.max_mention_per_cluster,
                                                         dtype=tf.float32))  # [k,max_mention_per_cluster]
    oracle_cluster_att_scores = tf.nn.softmax(oracle_cluster_att_scores, 1)
    oracle_cluster_emb = tf.reduce_sum(
      tf.gather(mention_emb, oracle_clusters) * tf.expand_dims(oracle_cluster_att_scores, 2), 1)  # [k,emb]
    oracle_cluster_mention_scores = tf.reduce_sum(
      tf.gather(mention_scores, oracle_clusters) * oracle_cluster_att_scores, 1)  # [k]

    top_span_range = tf.range(num_mentions)  # [k]
    cluster_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
    cluster_mask = cluster_offsets >= 1  # [k, k]

    if self.config["use_coarse_to_fine"]:
      fast_cluster_scores = tf.expand_dims(mention_scores,1) + tf.expand_dims(oracle_cluster_mention_scores,0) \
                            + tf.log(tf.to_float(cluster_mask))  # [k,k]
      with tf.variable_scope("coarse_to_fine_cluster_scoring", reuse=tf.AUTO_REUSE):
        c2f_mention_emb = tf.nn.dropout(util.projection(mention_emb , util.shape(mention_emb, -1)), self.dropout)  # [k,emb]
        c2f_cluster_emb = tf.nn.dropout(oracle_cluster_emb, self.dropout)  # [k,emb]

      fast_cluster_scores += tf.matmul(c2f_mention_emb, c2f_cluster_emb, transpose_b=True)  # [k,k]
      top_fast_cluster_scores, top_clusters = tf.nn.top_k(fast_cluster_scores, max_clusters, sorted=False)  # [k,max_ant]
      top_cluster_mask = util.batch_gather(cluster_mask,top_clusters)
      top_cluster_offsets = util.batch_gather(cluster_offsets,top_clusters)
    else:
      top_cluster_offsets = tf.tile(tf.expand_dims(tf.range(max_clusters)+1,0),[num_mentions,1]) #[k,max_ant]
      top_clusters = tf.expand_dims(tf.range(num_mentions),1) - top_cluster_offsets #[k,max_ant]
      top_cluster_mask = top_clusters >=0 #[k,max_ant]
      top_clusters = tf.maximum(top_clusters,0) #[k,max_ant]
      top_fast_cluster_scores = tf.expand_dims(mention_scores,1) + tf.gather(oracle_cluster_mention_scores, top_clusters)#[k,max_ant]
      top_fast_cluster_scores += tf.log(tf.to_float(top_cluster_mask))#[k,max_ant]

    top_cluster_width_bin = tf.gather(oracle_cluster_width_bin, top_clusters)
    top_cluster_emb = tf.gather(oracle_cluster_emb, top_clusters)
    top_cluster_indices = tf.gather(oracle_clusters, top_clusters)  # [k,max_ant,max_mention_per_cluster]
    top_individual_cluster_size = tf.gather(oracle_cluster_size, top_clusters)  # [k,max_ant]


    feature_emb_list = []
    with tf.variable_scope("feature_emb", reuse=tf.AUTO_REUSE):
      if self.config['use_metadata']:
        cl_speaker_ids = tf.gather(mention_speaker_ids, top_clusters)  # [k,max_ant]
        same_speaker = tf.equal(tf.expand_dims(mention_speaker_ids,1), cl_speaker_ids)  # [k,max_ant]
        speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                     tf.to_int32(same_speaker))  # [k,max_ant, emb]
        feature_emb_list.append(speaker_pair_emb)

        tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0),0),
                                  [num_mentions,max_clusters, 1])  # [k,max_ant, emb]
        feature_emb_list.append(tiled_genre_emb)

      if self.config['use_features']:
        mention_distance_bins = self.bucket_distance(top_cluster_offsets)  # [k,max_ant]
        mention_distance_emb = tf.gather(tf.get_variable("mention_distance_emb", [10, self.config["feature_size"]]),
                                         mention_distance_bins)  # [k,max_ant,emb]
        feature_emb_list.append(mention_distance_emb)

      if self.config['use_cluster_width']:
        # the cluster width features
        cluster_width_emb = tf.gather(tf.get_variable("cluster_width_emb", [9, self.config["feature_size"]]),
                                      top_cluster_width_bin)  # [k,max_ant,emb]
        feature_emb_list.append(cluster_width_emb)

    feature_emb = tf.concat(feature_emb_list, 2)  # [k,max_ant, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k,max_ant, emb]

    mention_emb_tiled = tf.tile(tf.expand_dims(mention_emb,1),
                               [1,max_clusters, 1])  # [k,max_ant, emb]
    similarity_emb = top_cluster_emb * mention_emb_tiled  # [k,max_ant, emb]

    pair_emb = tf.concat([mention_emb_tiled, top_cluster_emb, similarity_emb, feature_emb],
                         2)  # [k,max_ant, emb]

    with tf.variable_scope("iteration"):
      with tf.variable_scope("cluster_scoring", reuse=tf.AUTO_REUSE):
        cluster_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                   self.dropout)  # [k,max_ant, 1]
        cluster_scores = tf.squeeze(cluster_scores, 2)  # [k,max_ant]

    cluster_scores += top_fast_cluster_scores
    cluster_scores = tf.concat([mention_type_scores, cluster_scores], axis=1)  # [k, max_ant+n_types]


    _,_,_,predicted_antecedents, predicted_mention_types = \
      self.get_cluster_scores(mention_emb, mention_scores, mention_att_scores, mention_speaker_ids, genre_emb,
                              mention_type_scores,is_training)

    return cluster_scores, top_cluster_indices, top_individual_cluster_size, predicted_antecedents, predicted_mention_types




  def get_cluster_scores(self, mention_emb, mention_scores,mention_att_scores, mention_speaker_ids, genre_emb,mention_type_scores,is_training):
    num_mentions = util.shape(mention_emb, 0)
    mention_emb_size = util.shape(mention_emb, 1)
    if self.config["use_coarse_to_fine"]:
      max_top_clusters = tf.minimum(num_mentions, self.config["max_top_antecedents"])
      max_scan_clusters = num_mentions
    elif self.config["eval_unlimited_cluster"]:
      max_top_clusters =  tf.cond(is_training,lambda :tf.minimum(num_mentions, self.config["max_top_antecedents"]),lambda :num_mentions)
      max_scan_clusters = max_top_clusters
    else:
      max_top_clusters =  tf.minimum(num_mentions, self.config["max_top_antecedents"])
      max_scan_clusters = max_top_clusters
    max_mention_per_cluster = self.max_mention_per_cluster

    init_mention_sid = tf.range(0, num_mentions, dtype=tf.int32)
    init_cluster_size = tf.zeros([max_scan_clusters], dtype=tf.int32)
    init_cluster_last_mention = tf.zeros([max_scan_clusters], dtype=tf.int32)
    init_cluster_len = tf.constant(0, dtype=tf.int32)
    init_cluster_scores = tf.zeros([max_scan_clusters + self.n_types])
    init_cluster_indices = tf.zeros([max_scan_clusters, max_mention_per_cluster], dtype=tf.int32)
    init_cluster_emb = tf.zeros([max_scan_clusters, mention_emb_size])
    init_cluster_mention_scores = tf.zeros([max_scan_clusters])
    init_cluster_sid = tf.zeros([max_scan_clusters], dtype=tf.int32)

    def _cluster_scan(pre, input):
      m_emb, m_init_sid, m_score, m_type_score = input
      dim_m_emb, dim_m_init_sid, dim_m_score= \
        tf.expand_dims(m_emb, 0), tf.expand_dims(m_init_sid, 0) \
          , tf.expand_dims(m_score, 0)
      cl_emb,cl_m_score, cl_indices, cl_size, cl_sid, cl_last_m, cl_len, _, _, _, _, _ = pre


      fast_cluster_scores = dim_m_score + cl_m_score +tf.log(
        tf.sequence_mask(cl_len, max_scan_clusters, dtype=tf.float32))  # [k]

      if self.config["use_coarse_to_fine"]:
        with tf.variable_scope("coarse_to_fine_cluster_scoring",reuse=tf.AUTO_REUSE):
          c2f_mention_emb = tf.nn.dropout(util.projection(dim_m_emb,util.shape(dim_m_emb,-1)),self.dropout)#[1,emb]
          c2f_cluster_emb = tf.nn.dropout(cl_emb,self.dropout)#[k,emb]

        fast_cluster_scores+= tf.squeeze(tf.matmul(c2f_mention_emb,c2f_cluster_emb,transpose_b=True),0) #[k]
        top_fast_cluster_scores, top_clusters = tf.nn.top_k(fast_cluster_scores,max_top_clusters,sorted=False) #[max_ant]


        top_cl_emb = tf.gather(cl_emb,top_clusters)
        top_cl_indices = tf.gather(cl_indices,top_clusters)
        top_cl_size = tf.gather(cl_size,top_clusters)
        top_cl_last_m = tf.gather(cl_last_m,top_clusters)
      else:
        top_cl_emb = cl_emb
        top_cl_indices = cl_indices
        top_cl_size = cl_size
        top_cl_last_m = cl_last_m
        top_fast_cluster_scores = fast_cluster_scores



      feature_emb_list = []
      with tf.variable_scope("feature_emb", reuse=tf.AUTO_REUSE):
        if self.config['use_metadata']:
          cl_speaker_ids = tf.gather(mention_speaker_ids, top_cl_last_m)  # [max_ant]
          same_speaker = tf.equal(tf.gather(mention_speaker_ids, dim_m_init_sid), cl_speaker_ids)  # [max_ant]
          speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                       tf.to_int32(same_speaker))  # [max_ant, emb]
          feature_emb_list.append(speaker_pair_emb)

          tiled_genre_emb = tf.tile(tf.expand_dims(genre_emb, 0),
                                    [max_top_clusters, 1])  # [max_ant, emb]
          feature_emb_list.append(tiled_genre_emb)

        if self.config['use_features']:
          mention_distance = dim_m_init_sid - top_cl_last_m  # [max_ant]
          mention_distance_bins = coref_ops.distance_bins(mention_distance)  # [max_ant]
          mention_distance_bins.set_shape([None])
          mention_distance_emb = tf.gather(tf.get_variable("mention_distance_emb", [10, self.config["feature_size"]]),
                                           mention_distance_bins)  # [max_ant]
          feature_emb_list.append(mention_distance_emb)

        if self.config['use_cluster_width']:
          #the cluster width features
          cluster_width_bins = coref_ops.cluster_width_bins(top_cl_size)
          cluster_width_bins.set_shape([None])
          cluster_width_emb = tf.gather(tf.get_variable("cluster_width_emb", [9, self.config["feature_size"]]),
                                        cluster_width_bins)  # [max_ant]
          feature_emb_list.append(cluster_width_emb)

      feature_emb = tf.concat(feature_emb_list, 1)  # [max_ant, emb]
      feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [max_ant, emb]

      m_emb_tiled = tf.tile(dim_m_emb,
                                 [max_top_clusters, 1])  # [max_ant, emb]
      similarity_emb = top_cl_emb * m_emb_tiled  # [max_ant, emb]

      pair_emb = tf.concat([m_emb_tiled, top_cl_emb, similarity_emb, feature_emb],
                           1)  # [max_ant, emb]


      with tf.variable_scope("iteration"):
        with tf.variable_scope("cluster_scoring", reuse=tf.AUTO_REUSE):
          cluster_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                     self.dropout)  # [max_ant, 1]
          cluster_scores = tf.squeeze(cluster_scores, 1)  # [max_ant]

      cluster_scores += top_fast_cluster_scores
      cluster_scores = tf.concat([m_type_score, cluster_scores], axis=0)  # [max_ant+n_types]
      weighted_cluster_scores = tf.nn.softmax(cluster_scores, axis=0)  # [max_ant +n_types]

      step = tf.argmax(cluster_scores, output_type=tf.int32)
      step_weight = tf.gather(weighted_cluster_scores,step)

      def _exclude():
        return cl_emb,cl_m_score,cl_indices,cl_size,cl_sid,cl_last_m, cl_len,-1,step

      def _dn():
        indices = tf.range(0, max_scan_clusters, dtype=tf.int32)
        re_cl_emb = tf.gather(tf.concat([dim_m_emb, cl_emb], axis=0), indices)
        re_cl_m_score = tf.gather(tf.concat([dim_m_score, cl_m_score], axis=0), indices)
        step_cl_indices = tf.concat([dim_m_init_sid, tf.zeros([max_mention_per_cluster - 1], dtype=tf.int32)], axis=0)
        re_cl_indices = tf.gather(tf.concat([tf.expand_dims(step_cl_indices, axis=0), cl_indices], axis=0), indices)
        re_cl_size = tf.gather(tf.concat([tf.ones([1], dtype=tf.int32), cl_size], axis=0), indices)
        re_cl_sid = tf.gather(tf.concat([dim_m_init_sid, cl_sid], axis=0), indices) if self.config['use_history_clusters'] else cl_sid #cl_sid only used when using history clusters
        re_cl_last_m = tf.gather(tf.concat([dim_m_init_sid, cl_last_m], axis=0), indices)
        re_cl_len = tf.minimum(max_scan_clusters, cl_len + 1)
        re_ant = -1
        return re_cl_emb, re_cl_m_score, re_cl_indices, re_cl_size, re_cl_sid, re_cl_last_m, re_cl_len, re_ant,step

      def _do():
        if self.config['use_history_clusters']:
          _step = step - self.n_types
          if self.config["use_coarse_to_fine"]:
            _step = top_clusters[_step]
          step_sid = cl_sid[_step]
          act_step = tf.argmax(tf.to_int32(tf.equal(cl_sid, step_sid)), output_type=tf.int32)  # link to latest cluster
          indices = tf.range(0, max_scan_clusters, dtype=tf.int32)
        else:
          act_step = step - self.n_types
          if self.config["use_coarse_to_fine"]:
            act_step = top_clusters[act_step]
          step_sid = cl_sid[act_step]
          indices = tf.concat([tf.range(0, act_step + 1), tf.range(act_step + 2, max_scan_clusters + 1)], axis=0)

        step_cl_size = tf.minimum(max_mention_per_cluster, tf.gather(cl_size, act_step) + 1)
        step_cl_indices = tf.concat([tf.gather(tf.gather(cl_indices, act_step), tf.range(0, step_cl_size - 1)),
                                     dim_m_init_sid,
                                     tf.zeros([max_mention_per_cluster - step_cl_size], dtype=tf.int32)],
                                    axis=0)  # [max_mention_per_cluster]

        if self.config["use_cluster_position_emb"]:
          step_cl_indices_with_cluster_position = tf.concat(
            [tf.expand_dims(step_cl_indices, 1), tf.expand_dims(tf.range(0, max_mention_per_cluster), 1)],
            1)  # [max_mention_per_cluster,2]
          step_att_scores = tf.gather_nd(mention_att_scores,
                                         step_cl_indices_with_cluster_position)  # [max_mention_per_cluster]
        else:
          step_att_scores = tf.gather(mention_att_scores, step_cl_indices)

        att = tf.nn.softmax(
          step_att_scores + tf.log(tf.sequence_mask(step_cl_size, max_mention_per_cluster, dtype=tf.float32)), axis=0)
        step_cl_emb = tf.reduce_sum(tf.gather(mention_emb, step_cl_indices) * tf.expand_dims(att, axis=1),
                                    axis=0)
        step_cl_m_score = tf.reduce_sum(tf.gather(mention_scores, step_cl_indices) * att,
                                        axis=0)

        re_cl_emb = tf.gather(tf.concat([tf.expand_dims(step_cl_emb,axis=0), cl_emb], axis=0), indices)
        re_cl_m_score = tf.gather(tf.concat([tf.expand_dims(step_cl_m_score,axis=0), cl_m_score], axis=0), indices)
        re_cl_indices = tf.gather(tf.concat([tf.expand_dims(step_cl_indices, axis=0), cl_indices], axis=0), indices)
        re_cl_size = tf.gather(tf.concat([[step_cl_size], cl_size], axis=0), indices)
        re_cl_sid = tf.gather(tf.concat([tf.expand_dims(step_sid, axis=0), cl_sid], axis=0), indices) if self.config['use_history_clusters'] else cl_sid
        re_cl_last_m = tf.gather(tf.concat([dim_m_init_sid, cl_last_m], axis=0), indices)
        re_cl_len = tf.minimum(max_scan_clusters, cl_len + 1) if self.config['use_history_clusters'] else cl_len
        re_ant = tf.gather(cl_last_m, act_step)
        return re_cl_emb, re_cl_m_score, re_cl_indices, re_cl_size, re_cl_sid, re_cl_last_m, re_cl_len, re_ant,-1

      prefiltering_threshold = self.config['prefiltering_threshold']
      re_cl_emb,re_cl_m_score, re_cl_indices, re_cl_size, re_cl_sid, re_cl_last_m, re_cl_len, re_ant, re_m_type = \
        tf.cond(tf.less(step, self.n_types), lambda: tf.cond(tf.logical_and(tf.less(step, self.n_types - 1), tf.greater(step_weight, prefiltering_threshold)), _exclude,_dn), _do)

      re_cl_emb.set_shape([None, mention_emb_size])
      re_cl_m_score.set_shape([None])
      re_cl_indices.set_shape([None, None])
      re_cl_size.set_shape([None])
      re_cl_sid.set_shape([None])
      re_cl_last_m.set_shape([None])
      re_cl_len.set_shape([])
      re_ant.set_shape([])
      re_m_type.set_shape([])
      cluster_scores.set_shape([None])
      return re_cl_emb, re_cl_m_score, re_cl_indices, re_cl_size, re_cl_sid, re_cl_last_m, re_cl_len, re_ant, re_m_type, cluster_scores, top_cl_indices, top_cl_size

    _, _, _,_, _, _, _,  predicted_antecedents, predicted_mention_types, cluster_scores, cluster_indices, individual_cluster_size = tf.scan(
        _cluster_scan,
      (mention_emb, init_mention_sid, mention_scores, mention_type_scores),
      initializer=(
        init_cluster_emb,init_cluster_mention_scores,
        init_cluster_indices, init_cluster_size,
        init_cluster_sid, init_cluster_last_mention,
        init_cluster_len, 0,0, init_cluster_scores,
        init_cluster_indices, init_cluster_size
      ),swap_memory=True)
    cluster_indices, individual_cluster_size, predicted_antecedents,predicted_mention_types, cluster_scores = \
      tf.stack(cluster_indices), tf.stack(individual_cluster_size), \
      tf.stack(predicted_antecedents), tf.stack(predicted_mention_types), tf.stack(cluster_scores)

    return cluster_scores, cluster_indices, individual_cluster_size, predicted_antecedents, predicted_mention_types



  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    predicted_mention_types = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1)):
      if index < self.n_types:
        predicted_antecedents.append(-1)
        predicted_mention_types.append(index)
      else:
        predicted_antecedents.append(antecedents[i, index - self.n_types])
        predicted_mention_types.append(-1)
    return predicted_antecedents, predicted_mention_types

  def get_predicted_clusters_with_nr_singleton(self, mention_starts, mention_ends, predicted_antecedents,
                                               predicted_mention_types):
    mention_to_predicted = {}
    predicted_clusters = []
    predicted_mention_cluster_types = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(mention_starts[predicted_index]), int(mention_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster
        predicted_mention_cluster_types.append(
          (predicted_antecedent[0], predicted_antecedent[1], predicted_cluster, 'new'))

      mention = (int(mention_starts[i]), int(mention_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster
      predicted_mention_cluster_types.append((mention[0], mention[1], predicted_cluster, 'old'))

    curr_sid = len(predicted_clusters)

    for s, e, type in zip(mention_starts, mention_ends, predicted_mention_types):
      mention = (int(s), int(e))
      if type == 0 or mention in mention_to_predicted:
        continue

      label = 'non_referring' if type < self.n_types - 1 else 'new'

      predicted_cluster = curr_sid
      curr_sid += 1
      predicted_mention_cluster_types.append((mention[0], mention[1], predicted_cluster, label))

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

    return predicted_clusters, mention_to_predicted, predicted_mention_cluster_types

  def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, predicted_mention_types, gold_clusters, evaluator):
    if self.crac_doc:
      gold_clusters = [tuple(tuple((s,e)) for (s,e,t,_) in gc) for gc in gold_clusters if len(gc) >1] #exclude non-referring and singleton
    else:
      gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    if self.crac_doc:
      predicted_clusters, mention_to_predicted, predicted_mention_cluster_types = self.get_predicted_clusters_with_nr_singleton(
        top_span_starts, top_span_ends, predicted_antecedents, predicted_mention_types)
      evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
      return predicted_mention_cluster_types
    else:
      predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                             predicted_antecedents)
      evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
      return predicted_clusters


  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, official_eval=False, official_stdout=False):
    if official_stdout or self.crac_doc:  # for crac the official_eval is compulsory
      official_eval = True
    self.load_eval_data()

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      top_span_starts, top_span_ends, predicted_antecedents, predicted_mention_types = session.run(self.predictions, feed_dict=feed_dict)

      coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, predicted_mention_types, example["clusters"], coref_evaluator)
      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    summary_dict = {}


    if official_eval:
      if self.crac_doc:
        predicted_path = self.config["out_dir"]
        average_f1, average_conll_f1, average_nr_f1 = crac.eval_crac(self.config["conll_eval_path"], coref_predictions,
                                                                     predicted_path, official_stdout)
        summary_dict["Average F1 (85%conll+15%nr)"] = average_f1
        summary_dict["Average F1 (conll)"] = average_conll_f1
        summary_dict["Average F1 (nr)"] = average_nr_f1
        print ("Average F1 (85%conll+15%nr): {:.2f}%".format(average_f1))
        print ("Average F1 (conll): {:.2f}%".format(average_conll_f1))
        print ("Average F1 (nr): {:.2f}%".format(average_nr_f1))
      else:
        eval_file = self.config["conll_eval_path"]
        eval_file = eval_file[eval_file.rfind('/') + 1:] if '/' in eval_file else eval_file
        predicted_path = os.path.join(self.config["out_dir"], eval_file)
        conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, predicted_path,
                                             official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"] = average_f1
        print ("Average F1 (conll): {:.2f}%".format(average_f1))

    p, r, f = coref_evaluator.get_prf()
    summary_dict["Average F1 (py)"] = f
    print ("Average F1 (py): {:.2f}%".format(f * 100))
    summary_dict["Average precision (py)"] = p
    print ("Average precision (py): {:.2f}%".format(p * 100))
    summary_dict["Average recall (py)"] = r
    print ("Average recall (py): {:.2f}%".format(r * 100))
    average_f1 = f * 100 if not official_eval else average_f1

    return util.make_summary(summary_dict), average_f1






