#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()

  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]
  #evaluate unlimited cluster only used in the final evaluation
  config["eval_unlimited_cluster"] = False
  with_split_antecedent = config["with_split_antecedent"]
  model = cm.CorefModel(config)
  saver = tf.train.Saver(max_to_keep=1)

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_step = config["max_step"]
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  session_config.allow_soft_placement = True

  max_f1 = 0
  max_point = 0

  with tf.Session(config=session_config) as session:

    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)

    initial_time = time.time()

    tf_global_step = 0

    while tf_global_step < max_step:
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
      accumulated_loss += tf_loss

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print("Coref: [{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model.ckpt"), global_step=tf_global_step)
        eval_summary, eval_f1 = model.evaluate(session,mode='coref')

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          max_point = tf_global_step
          util.copy_checkpoint(os.path.join(log_dir, "model.ckpt-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        print("Coref: [{}] evaL_f1={:.2f}, max_f1={:.2f} from step {}".format(tf_global_step, eval_f1, max_f1,max_point))

    if with_split_antecedent:
      util.copy_checkpoint(os.path.join(log_dir, "model.max.ckpt"),
                           os.path.join(log_dir, "model.max.coref.ckpt"))

      print('\n\nCoref training finished!\n\n Starting training split antecedent.')
      eval_frequency = 500  # more freq
      accumulated_loss = 0
      max_f1 = 0
      model.restore(session)
      model.train_mode = 'split_antecedent'
      session.run(model.reset_global_step)
      tf_global_step = 0
      initial_time = time.time()
      while tf_global_step < max_step * 0.2:
        tf_loss, tf_global_step, _ = session.run([model.split_antecedent_loss, model.global_step, model.split_antecedent_train_op])
        accumulated_loss += tf_loss

        if tf_global_step % report_frequency == 0:
          total_time = time.time() - initial_time
          steps_per_second = tf_global_step / total_time

          average_loss = accumulated_loss / report_frequency
          print(
            "Split antecedent - [{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
          writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
          accumulated_loss = 0.0

        if tf_global_step % eval_frequency == 0:
          saver.save(session, os.path.join(log_dir, "model.ckpt"), global_step=tf_global_step)
          eval_summary, eval_f1 = model.evaluate(session, mode="split_antecedent")

          if eval_f1 > max_f1:
            max_f1 = eval_f1
            max_point = tf_global_step
            util.copy_checkpoint(os.path.join(log_dir, "model.ckpt-{}".format(tf_global_step)),
                                 os.path.join(log_dir, "model.max.ckpt"))

          writer.add_summary(eval_summary, tf_global_step)
          writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

          print("Split antecedent - [{}] evaL_f1={:.2f}, max_f1={:.2f} from step {}".format(tf_global_step, eval_f1, max_f1,
                                                                              max_point))