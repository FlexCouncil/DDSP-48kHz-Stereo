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
"""Library of Trainer objects that define training step and wrap optimizer."""

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
        """Constructor."""
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
        	self.optimizer = tf.keras.optimizers.Adam(lr_schedule)

    def save(self, save_dir):
        """Saves model and optimizer to a checkpoint."""
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

        restore_keys = self.restore_keys if restore_keys is None else restore_keys
        if restore_keys is None:
            model = self.model
            logging.info('Trainer restoring the full model')
        else:
            restore_dict = {k: getattr(self.model, k) for k in restore_keys}
            model = tf.train.Checkpoint(**restore_dict)
            logging.info('Trainer restoring model subcomponents:')
            for k, v in restore_dict.items():
                logging.info('Restoring %s: %s', k, v)

        checkpoint = tf.train.Checkpoint(model=model, optimizer=self.optimizer)
        latest_checkpoint = train_util.get_latest_chekpoint(checkpoint_path)
        if latest_checkpoint is not None:
            with self.strategy.scope():
                status = checkpoint.restore(latest_checkpoint)
                if restore_keys is None:
                    status.assert_consumed()
                else:
                    status.expect_partial()
                logging.info('Loaded checkpoint %s', latest_checkpoint)
            logging.info('Loading model took %.1f seconds', time.time() - start_time)
        else:
            logging.info('No checkpoint found, skipping restoration.')

    @property
    def step(self):
        """The number of training steps completed."""
        return self.optimizer.iterations

    def build(self, batch):
        """Build the model by running a batch through it."""
        logging.info('Building the model...')
        with self.strategy.scope():
            _ = self.model(batch)
        self.model.summary()

    def distribute_dataset(self, dataset):
        """Create a distributed dataset."""
        if isinstance(dataset, tf.data.Dataset):
            return self.strategy.experimental_distribute_dataset(dataset)
        else:
            return dataset

    @tf.function
    def train_step(self, inputs):
        """Training step."""
        if hasattr(inputs, '__next__'):
            batch = next(inputs)
        else:
            batch = inputs

        def step_fn(batch):
            with tf.GradientTape() as tape:
                _, losses = self.model(batch, return_losses=True, training=True)
                total_loss = sum(losses.values())

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            return losses

        # Use strategy.run to ensure we're in the replica context
        per_replica_losses = self.strategy.run(step_fn, args=(batch,))

        # Reduce losses across replicas
        loss_dict = {
            k: self.strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
            for k, v in per_replica_losses.items()
        }
        return loss_dict