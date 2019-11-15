from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

coref_op_library = tf.load_op_library("./coref_kernels.so")

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")

gold_scores = coref_op_library.gold_scores
tf.NotDifferentiable("GoldScores")

distance_bins = coref_op_library.distance_bins
tf.NotDifferentiable("DistanceBins")

cluster_width_bins = coref_op_library.cluster_width_bins
tf.NotDifferentiable("ClusterWidthBins")

extract_antecedent_labels = coref_op_library.extract_antecedent_labels
tf.NotDifferentiable("ExtractAntecedentLabels")

oracle_clusters = coref_op_library.oracle_clusters
tf.NotDifferentiable("OracleClusters")
