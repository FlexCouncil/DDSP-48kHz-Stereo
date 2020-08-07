# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified from the original

# Lint as: python3
"""Library of Trainer objects that define traning step and wrap optimizer."""

import time

from absl import logging
from training import train_util
import gin
import tensorflow.compat.v2 as tf


@gin.configurable
class Trainer(object):
  """Class to bind an optimizer, model, strategy, and training step function."""

  def __init__(self,
               model,
               strategy,
               checkpoints_to_keep=100,
               learning_rate=0.001,
               lr_decay_steps=10000,
               lr_decay_rate=0.98,
               grad_clip_norm=3.0,
               restore_keys=None):
    """Constructor.

    Args:
      model: Model to train.
      strategy: A distribution strategy.
      checkpoints_to_keep: Max number of checkpoints before deleting oldest.
      learning_rate: Scalar initial learning rate.
      lr_decay_steps: Exponential decay timescale.
      lr_decay_rate: Exponential decay magnitude.
      grad_clip_norm: Norm level by which to clip gradients.
      restore_keys: List of names of model properties to restore. If no keys are
        passed, restore the whole model.
    """
    self.model = model
    self.strategy = strategy
    self.checkpoints_to_keep = checkpoints_to_keep
    self.grad_clip_norm = grad_clip_norm
    self.restore_keys = restore_keys

    # Create an optimizer.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate)

    with self.strategy.scope():
      optimizer = tf.keras.optimizers.Adam(lr_schedule)
      self.optimizer = optimizer

  def save(self, save_dir):
    """Saves model and optimizer to a checkpoint."""
    # Saving weights in checkpoint format because saved_model requires
    # handling variable batch size, which some synths and effects can't.
    start_time = time.time()
    checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=save_dir, max_to_keep=self.checkpoints_to_keep)
    step = self.step.numpy()
    manager.save(checkpoint_number=step)
    logging.info('Saved checkpoint to %s at step %s', save_dir, step)
    logging.info('Saving model took %.1f seconds', time.time() - start_time)

  def restore(self, checkpoint_path, restore_keys=None):
    """Restore model and optimizer from a checkpoint if it exists."""
    logging.info('Restoring from checkpoint...')
    start_time = time.time()

    # Prefer function args over object properties.
    restore_keys = self.restore_keys if restore_keys is None else restore_keys
    if restore_keys is None:
      # If no keys are passed, restore the whole model.
      model = self.model
      logging.info('Trainer restoring the full model')
    else:
      # Restore only sub-modules by building a new subgraph.
      restore_dict = {k: getattr(self.model, k) for k in restore_keys}
      model = tf.train.Checkpoint(**restore_dict)

      logging.info('Trainer restoring model subcomponents:')
      for k, v in restore_dict.items():
        log_str = 'Restoring {}: {}'.format(k, v)
        logging.info(log_str)

    # Restore from latest checkpoint.
    checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
    latest_checkpoint = train_util.get_latest_chekpoint(checkpoint_path)
    if latest_checkpoint is not None:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      with self.strategy.scope():
        if restore_keys is None:
          checkpoint.restore(latest_checkpoint)
        else:
          checkpoint.restore(latest_checkpoint).expect_partial()
        logging.info('Loaded checkpoint %s', latest_checkpoint)
      logging.info('Loading model took %.1f seconds', time.time() - start_time)
    else:
      logging.info('No checkpoint, skipping.')

  @property
  def step(self):
    """The number of training steps completed."""
    return self.optimizer.iterations

  def psum(self, x, axis=None):
    """Sum across processors."""
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=axis)

  def run(self, fn, *args, **kwargs):
    """Distribute and run function on processors."""
    return self.strategy.run(fn, args=args, kwargs=kwargs)

  def build(self, batch):
    """Build the model by running a distributed batch through it."""
    logging.info('Building the model...')
    _ = self.run(tf.function(self.model.__call__), batch)
    self.model.summary()

  def distribute_dataset(self, dataset):
    """Create a distributed dataset."""
    if isinstance(dataset, tf.data.Dataset):
      return self.strategy.experimental_distribute_dataset(dataset)
    else:
      return dataset

  @tf.function
  def train_step(self, inputs):
    """Distributed training step."""
    # Wrap iterator in tf.function, slight speedup passing in iter vs batch.
    batch = next(inputs) if hasattr(inputs, '__next__') else inputs
    losses = self.run(self.step_fn, batch)
    # Add up the scalar losses across replicas.
    n_replicas = self.strategy.num_replicas_in_sync
    return {k: self.psum(v, axis=None) / n_replicas for k, v in losses.items()}

  @tf.function
  def step_fn(self, batch):
    """Per-Replica training step."""
    with tf.GradientTape(persistent=True) as tape:
      _, losses = self.model(batch, return_losses=True, training=True)
    # Clip and apply gradients.
    tvars = self.model.trainable_variables
    m_vars1 = [var for var in tvars if '/normalize/' in var.name]
    m_vars2 = [var for var in tvars if '/gru_cell/' in var.name]
    m_vars3 = [var for var in tvars if '/gru_cell_3/' in var.name]
    m_vars4 = [var for var in tvars if 'rnn_encoder/dense/' in var.name]
    m_vars5 = [var for var in tvars if '/fc_stack/' in var.name]
    m_vars6 = [var for var in tvars if '/fc_stack_1/' in var.name]
    m_vars7 = [var for var in tvars if '/fc_stack_2/' in var.name]
    m_vars8 = [var for var in tvars if '/fc_stack_9/' in var.name]
    m_vars9 = [var for var in tvars if 'processor_group/' in var.name]
    m_vars10 = [var for var in tvars if 'rnn_fc_decoder/dense/' in var.name]
    l_vars1 = [var for var in tvars if '/normalize_1/' in var.name]
    l_vars2 = [var for var in tvars if '/gru_cell_1/' in var.name]
    l_vars3 = [var for var in tvars if '/gru_cell_4/' in var.name]
    l_vars4 = [var for var in tvars if 'rnn_encoder/dense_1/' in var.name]
    l_vars5 = [var for var in tvars if '/fc_stack_3/' in var.name]
    l_vars6 = [var for var in tvars if '/fc_stack_4/' in var.name]
    l_vars7 = [var for var in tvars if '/fc_stack_5/' in var.name]
    l_vars8 = [var for var in tvars if '/fc_stack_10/' in var.name]
    l_vars9 = [var for var in tvars if 'processor_group_1/' in var.name]
    l_vars10 = [var for var in tvars if '/rnn_fc_decoder/dense_1/' in var.name]
    r_vars1 = [var for var in tvars if '/normalize_2/' in var.name]
    r_vars2 = [var for var in tvars if '/gru_cell_2/' in var.name]
    r_vars3 = [var for var in tvars if '/gru_cell_5/' in var.name]
    r_vars4 = [var for var in tvars if 'rnn_encoder/dense_2/' in var.name]
    r_vars5 = [var for var in tvars if '/fc_stack_6/' in var.name]
    r_vars6 = [var for var in tvars if '/fc_stack_7/' in var.name]
    r_vars7 = [var for var in tvars if '/fc_stack_8/' in var.name]
    r_vars8 = [var for var in tvars if '/fc_stack_11/' in var.name]
    r_vars9 = [var for var in tvars if 'processor_group_2/' in var.name]
    r_vars10 = [var for var in tvars if '/rnn_fc_decoder/dense_2/' in var.name]
    gradsM1 = tape.gradient(losses['spectral_loss_mono'], m_vars1)
    gradsM2 = tape.gradient(losses['spectral_loss_mono'], m_vars2)
    gradsM3 = tape.gradient(losses['spectral_loss_mono'], m_vars3)
    gradsM4 = tape.gradient(losses['spectral_loss_mono'], m_vars4)
    gradsM5 = tape.gradient(losses['spectral_loss_mono'], m_vars5)
    gradsM6 = tape.gradient(losses['spectral_loss_mono'], m_vars6)
    gradsM7 = tape.gradient(losses['spectral_loss_mono'], m_vars7)
    gradsM8 = tape.gradient(losses['spectral_loss_mono'], m_vars8)
    gradsM9 = tape.gradient(losses['spectral_loss_mono'], m_vars9)
    gradsM10 = tape.gradient(losses['spectral_loss_mono'], m_vars10)
    gradsL1 = tape.gradient(losses['spectral_loss_left'], l_vars1)
    gradsL2 = tape.gradient(losses['spectral_loss_left'], l_vars2)
    gradsL3 = tape.gradient(losses['spectral_loss_left'], l_vars3)
    gradsL4 = tape.gradient(losses['spectral_loss_left'], l_vars4)
    gradsL5 = tape.gradient(losses['spectral_loss_left'], l_vars5)
    gradsL6 = tape.gradient(losses['spectral_loss_left'], l_vars6)
    gradsL7 = tape.gradient(losses['spectral_loss_left'], l_vars7)
    gradsL8 = tape.gradient(losses['spectral_loss_left'], l_vars8)
    gradsL9 = tape.gradient(losses['spectral_loss_left'], l_vars9)
    gradsL10 = tape.gradient(losses['spectral_loss_left'], l_vars10)
    gradsR1 = tape.gradient(losses['spectral_loss_right'], r_vars1)
    gradsR2 = tape.gradient(losses['spectral_loss_right'], r_vars2)
    gradsR3 = tape.gradient(losses['spectral_loss_right'], r_vars3)
    gradsR4 = tape.gradient(losses['spectral_loss_right'], r_vars4)
    gradsR5 = tape.gradient(losses['spectral_loss_right'], r_vars5)
    gradsR6 = tape.gradient(losses['spectral_loss_right'], r_vars6)
    gradsR7 = tape.gradient(losses['spectral_loss_right'], r_vars7)
    gradsR8 = tape.gradient(losses['spectral_loss_right'], r_vars8)
    gradsR9 = tape.gradient(losses['spectral_loss_right'], r_vars9)
    gradsR10 = tape.gradient(losses['spectral_loss_right'], r_vars10)
    gradsM1, _ = tf.clip_by_global_norm(gradsM1, self.grad_clip_norm)
    gradsM2, _ = tf.clip_by_global_norm(gradsM2, self.grad_clip_norm)
    gradsM3, _ = tf.clip_by_global_norm(gradsM3, self.grad_clip_norm)
    gradsM4, _ = tf.clip_by_global_norm(gradsM4, self.grad_clip_norm)
    gradsM5, _ = tf.clip_by_global_norm(gradsM5, self.grad_clip_norm)
    gradsM6, _ = tf.clip_by_global_norm(gradsM6, self.grad_clip_norm)
    gradsM7, _ = tf.clip_by_global_norm(gradsM7, self.grad_clip_norm)
    gradsM8, _ = tf.clip_by_global_norm(gradsM8, self.grad_clip_norm)
    gradsM9, _ = tf.clip_by_global_norm(gradsM9, self.grad_clip_norm)
    gradsM10, _ = tf.clip_by_global_norm(gradsM10, self.grad_clip_norm)
    gradsL1, _ = tf.clip_by_global_norm(gradsL1, self.grad_clip_norm)
    gradsL2, _ = tf.clip_by_global_norm(gradsL2, self.grad_clip_norm)
    gradsL3, _ = tf.clip_by_global_norm(gradsL3, self.grad_clip_norm)
    gradsL4, _ = tf.clip_by_global_norm(gradsL4, self.grad_clip_norm)
    gradsL5, _ = tf.clip_by_global_norm(gradsL5, self.grad_clip_norm)
    gradsL6, _ = tf.clip_by_global_norm(gradsL6, self.grad_clip_norm)
    gradsL7, _ = tf.clip_by_global_norm(gradsL7, self.grad_clip_norm)
    gradsL8, _ = tf.clip_by_global_norm(gradsL8, self.grad_clip_norm)
    gradsL9, _ = tf.clip_by_global_norm(gradsL9, self.grad_clip_norm)
    gradsL10, _ = tf.clip_by_global_norm(gradsL10, self.grad_clip_norm)
    gradsR1, _ = tf.clip_by_global_norm(gradsR1, self.grad_clip_norm)
    gradsR2, _ = tf.clip_by_global_norm(gradsR2, self.grad_clip_norm)
    gradsR3, _ = tf.clip_by_global_norm(gradsR3, self.grad_clip_norm)
    gradsR4, _ = tf.clip_by_global_norm(gradsR4, self.grad_clip_norm)
    gradsR5, _ = tf.clip_by_global_norm(gradsR5, self.grad_clip_norm)
    gradsR6, _ = tf.clip_by_global_norm(gradsR6, self.grad_clip_norm)
    gradsR7, _ = tf.clip_by_global_norm(gradsR7, self.grad_clip_norm)
    gradsR8, _ = tf.clip_by_global_norm(gradsR8, self.grad_clip_norm)
    gradsR9, _ = tf.clip_by_global_norm(gradsR9, self.grad_clip_norm)
    gradsR10, _ = tf.clip_by_global_norm(gradsR10, self.grad_clip_norm)
    self.optimizer.apply_gradients(zip(gradsM1, m_vars1))
    self.optimizer.apply_gradients(zip(gradsM2, m_vars2))
    self.optimizer.apply_gradients(zip(gradsM3, m_vars3))
    self.optimizer.apply_gradients(zip(gradsM4, m_vars4))
    self.optimizer.apply_gradients(zip(gradsM5, m_vars5))
    self.optimizer.apply_gradients(zip(gradsM6, m_vars6))
    self.optimizer.apply_gradients(zip(gradsM7, m_vars7))
    self.optimizer.apply_gradients(zip(gradsM8, m_vars8))
    self.optimizer.apply_gradients(zip(gradsM9, m_vars9))
    self.optimizer.apply_gradients(zip(gradsM10, m_vars10))
    self.optimizer.apply_gradients(zip(gradsL1, l_vars1))
    self.optimizer.apply_gradients(zip(gradsL2, l_vars2))
    self.optimizer.apply_gradients(zip(gradsL3, l_vars3))
    self.optimizer.apply_gradients(zip(gradsL4, l_vars4))
    self.optimizer.apply_gradients(zip(gradsL5, l_vars5))
    self.optimizer.apply_gradients(zip(gradsL6, l_vars6))
    self.optimizer.apply_gradients(zip(gradsL7, l_vars7))
    self.optimizer.apply_gradients(zip(gradsL8, l_vars8))
    self.optimizer.apply_gradients(zip(gradsL9, l_vars9))
    self.optimizer.apply_gradients(zip(gradsL10, l_vars10))
    self.optimizer.apply_gradients(zip(gradsR1, r_vars1))
    self.optimizer.apply_gradients(zip(gradsR2, r_vars2))
    self.optimizer.apply_gradients(zip(gradsR3, r_vars3))
    self.optimizer.apply_gradients(zip(gradsR4, r_vars4))
    self.optimizer.apply_gradients(zip(gradsR5, r_vars5))
    self.optimizer.apply_gradients(zip(gradsR6, r_vars6))
    self.optimizer.apply_gradients(zip(gradsR7, r_vars7))
    self.optimizer.apply_gradients(zip(gradsR8, r_vars8))
    self.optimizer.apply_gradients(zip(gradsR9, r_vars9))
    self.optimizer.apply_gradients(zip(gradsR10, r_vars10))
    return losses


