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
"""Model that outputs coefficeints of an additive synthesizer."""

import time

from typing import Dict, Sequence, Tuple, Text

from absl import logging
import core
import losses
import processors
import synths
import effects
from training import train_util
import gin
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


@gin.configurable
def get_model(model=gin.REQUIRED):
  """Gin configurable function get a 'global' model for use in ddsp_run.py.

  Convenience for using the same model in train(), evaluate(), and sample().
  Args:
    model: An instantiated model, such as 'models.Autoencoder()'.

  Returns:
    The 'global' model specifieed in the gin config.
  """
  return model


class Model(tf.keras.Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self, name='model'):
    super().__init__(name=name)
    self._losses_dict = {}

  def __call__(self, *args, return_losses=False, **kwargs):
    """Reset the losses dict on each call.

    Args:
      *args: Arguments passed on to call().
      return_losses: Return a dictionary of losses in addition to the call()
        function returns.
      **kwargs: Keyword arguments passed on to call().

    Returns:
      Function results if return_losses=False, else the function results
        and a dictionary of losses, {loss_name: loss_value}.
    """
    self._losses_dict = {}
    results = super().__call__(*args, **kwargs)
    if not return_losses:
      return results
    else:
      self._losses_dict['total_loss'] = tf.reduce_sum(
          list(self._losses_dict.values()))
      return results, self._losses_dict

  def restore(self, checkpoint_path):
    """Restore model and optimizer from a checkpoint."""
    start_time = time.time()
    latest_checkpoint = train_util.get_latest_chekpoint(checkpoint_path)
    if latest_checkpoint is not None:
      checkpoint = tf.train.Checkpoint(model=self)
      checkpoint.restore(latest_checkpoint).expect_partial()
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      logging.info('Loading model took %.1f seconds', time.time() - start_time)
    else:
      logging.info('Could not find checkpoint to load at %s, skipping.',
                   checkpoint_path)


@gin.configurable
class Autoencoder(Model):
  """Wrap the model function for dependency injection with gin."""
  # adapted for stereo

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               n_samples=192000,
               losses=losses.SpectralLoss(),
               name='autoencoder'):
    super().__init__(name=name)
    processor_groupM=processors.ProcessorGroup(dag=[(synths.Additive(n_samples=n_samples), ['amps', 'harmonic_distribution', 'f0_hzM']), (synths.FilteredNoise(n_samples=n_samples), ['noise_magnitudes']), (processors.Add(), ['filtered_noise/signal', 'additive/signal']), (effects.Reverb(), ['add/signal']),])
    processor_groupL=processors.ProcessorGroup(dag=[(synths.Additive(n_samples=n_samples), ['amps', 'harmonic_distribution', 'f0_hzL']), (synths.FilteredNoise(n_samples=n_samples), ['noise_magnitudes']), (processors.Add(), ['filtered_noise/signal', 'additive/signal']), (effects.Reverb(), ['add/signal']),])
    processor_groupR=processors.ProcessorGroup(dag=[(synths.Additive(n_samples=n_samples), ['amps', 'harmonic_distribution', 'f0_hzR']), (synths.FilteredNoise(n_samples=n_samples), ['noise_magnitudes']), (processors.Add(), ['filtered_noise/signal', 'additive/signal']), (effects.Reverb(), ['add/signal']),])
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_groupM = processor_groupM
    self.processor_groupL = processor_groupL
    self.processor_groupR = processor_groupR
    self.loss_objs = core.make_iterable(losses)
       

  def controls_to_audio(self, controls):
    return controls[self.processor_group.name]['signal']

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    conditioning = self.preprocessor(features, training=training)
    return conditioning if self.encoder is None else self.encoder(conditioning)

  def decode(self, conditioning, training=True):
    """Get generated audio by decoding than processing."""
    # adapted for stereo
    processor_inputsM = self.decoder(conditioning, mode='mono', training=training)
    processor_inputsL = self.decoder(conditioning, mode='left', training=training)
    processor_inputsR = self.decoder(conditioning, mode='right', training=training)
    processedM = self.processor_groupM(processor_inputsM)
    processedL = self.processor_groupL(processor_inputsL)
    processedR = self.processor_groupR(processor_inputsR)
    return processedM, processedL, processedR
    
  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    # adapted for stereo
    conditioning = self.encode(features, training=training)
    audio_genM, audio_genL, audio_genR = self.decode(conditioning, training=training)
    if training:
      for loss_obj in self.loss_objs:
        lossM = loss_obj(features['audioM'], audio_genM)
        lossL = loss_obj(features['audioL'], audio_genL)
        lossR = loss_obj(features['audioR'], audio_genR)
        self._losses_dict['spectral_loss_mono'] = lossM
        self._losses_dict['spectral_loss_left'] = lossL
        self._losses_dict['spectral_loss_right'] = lossR
    return audio_genM, audio_genL, audio_genR

  def get_controls(self, features, keys=None, training=False):
    """Returns specific processor_group controls."""
    conditioning = self.encode(features, training=training)
    processor_inputs = self.decoder(conditioning)
    controls = self.processor_group.get_controls(processor_inputs)
    # Also build on get_controls(), instead of just __call__().
    self.built = True
    # If wrapped in tf.function, only calculates keys of interest.
    return controls if keys is None else {k: controls[k] for k in keys}


