#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  config['conll_eval_path'] = config['conll_test_path']
  config['eval_path'] = config['test_path']
  config['lm_path'] = config['test_lm_path']
  config["out_dir"] = util.mkdirs(os.path.join(config["log_dir"],'test_out'))

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True

  model = cm.CorefModel(config)
  with tf.Session(config=session_config) as session:
    model.restore(session)
    model.evaluate(session, official_stdout=True)
