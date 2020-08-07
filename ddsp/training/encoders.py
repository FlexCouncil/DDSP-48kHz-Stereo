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
"""Library of encoder objects."""

import spectral_ops
from training import nn
import gin
import numpy as np
import tensorflow.compat.v2 as tf

from typing import Any, Dict, Text, TypeVar
Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)

tfkl = tf.keras.layers

gin.enter_interactive_mode()


# ------------------ Encoders --------------------------------------------------
class Encoder(tfkl.Layer):
  """Base class to implement any encoder.

  Users should override compute_z() to define the actual encoder structure.
  Optionally, set infer_f0 to True and override compute_f0.
  Hyper-parameters will be passed through the constructor.
  """

  def __init__(self, f0_encoder=None, name='encoder'):
    super().__init__(name=name)
    self.f0_encoder = f0_encoder

  def call(self, conditioning):
    """Updates conditioning with z and (optionally) f0."""
    # adapted for stereo
    if self.f0_encoder:
      # Use frequency conditioning created by the f0_encoder, not the dataset.
      # Overwrite `f0_scaled` and `f0_hz`. 'f0_scaled' is a value in [0, 1]
      # corresponding to midi values [0..127]
      conditioning['f0_scaledM'] = self.f0_encoder(conditioning)
      conditioning['f0_scaledL'] = self.f0_encoder(conditioning)
      conditioning['f0_scaledR'] = self.f0_encoder(conditioning)
      conditioning['f0_hzM'] = midi_to_hz(conditioning['f0_scaledM'] * 127.0)
      conditioning['f0_hzL'] = midi_to_hz(conditioning['f0_scaledL'] * 127.0)
      conditioning['f0_hzR'] = midi_to_hz(conditioning['f0_scaledR'] * 127.0)
    zM, zL, zR = self.compute_z(conditioning)
    time_steps = int(conditioning['f0_scaledM'].shape[1])
    conditioning['zM'] = self.expand_z(zM, time_steps)
    conditioning['zL'] = self.expand_z(zL, time_steps)
    conditioning['zR'] = self.expand_z(zR, time_steps)

    return conditioning

  def expand_z(self, z, time_steps):
    """Make sure z has same temporal resolution as other conditioning."""
    # Add time dim of z if necessary.
    if len(z.shape) == 2:
      z = z[:, tf.newaxis, :]
    # Expand time dim of z if necessary.
    z_time_steps = int(z.shape[1])
    if z_time_steps != time_steps:
      z = resample(z, time_steps)
    return z

  def compute_z(self, conditioning):
    """Takes in conditioning dictionary, returns a latent tensor z."""
    raise NotImplementedError


@gin.register
class MfccTimeDistributedRnnEncoder(Encoder):
  """Use MFCCs as latent variables, distribute across timesteps."""
  # adapted for stereo

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               z_dims=16,
               z_time_steps=125,
               f0_encoder=None,
               name='mfcc_time_distrbuted_rnn_encoder'):
    super().__init__(f0_encoder=f0_encoder, name=name)
    if z_time_steps not in [63, 125, 250, 500, 1000]:
      raise ValueError(
          '`z_time_steps` currently limited to 63,125,250,500 and 1000')
    self.z_audio_spec = {
        '63': {
            'fft_size': 2048,
            'overlap': 0.5
        },
        '125': {
            'fft_size': 1024,
            'overlap': 0.5
        },
        '250': {
            'fft_size': 1024,
            'overlap': 0.75
        },
        '500': {
            'fft_size': 512,
            'overlap': 0.75
        },
        '1000': {
            'fft_size': 256,
            'overlap': 0.75
        }
    }
    self.fft_size = self.z_audio_spec[str(z_time_steps)]['fft_size']
    self.overlap = self.z_audio_spec[str(z_time_steps)]['overlap']

    # Layers.
    self.z_normM = nn.Normalize('instance')
    self.z_normL = nn.Normalize('instance')
    self.z_normR = nn.Normalize('instance')
    self.rnnM = nn.rnn(rnn_channels, rnn_type)
    self.rnnL = nn.rnn(rnn_channels, rnn_type)
    self.rnnR = nn.rnn(rnn_channels, rnn_type)
    self.dense_outM = nn.dense(z_dims)
    self.dense_outL = nn.dense(z_dims)
    self.dense_outR = nn.dense(z_dims)

  def compute_z(self, conditioning):
    #mono
    mfccsM = spectral_ops.compute_mfcc(
        conditioning['audioM'],
        lo_hz=20.0,
        hi_hz=24000.0,
        fft_size=self.fft_size,
        mel_bins=128,
        mfcc_bins=30,
        overlap=self.overlap,
        pad_end=True)
        
    #left 
    mfccsL = spectral_ops.compute_mfcc(
        conditioning['audioL'],
        lo_hz=20.0,
        hi_hz=24000.0,
        fft_size=self.fft_size,
        mel_bins=128,
        mfcc_bins=30,
        overlap=self.overlap,
        pad_end=True)
        
    #right
    mfccsR = spectral_ops.compute_mfcc(
        conditioning['audioR'],
        lo_hz=20.0,
        hi_hz=24000.0,
        fft_size=self.fft_size,
        mel_bins=128,
        mfcc_bins=30,
        overlap=self.overlap,
        pad_end=True)

    #mono
    # Normalize.
    zM = self.z_normM(mfccsM[:, :, tf.newaxis, :])[:, :, 0, :]
    # Run an RNN over the latents.
    zM = self.rnnM(zM)
    # Bounce down to compressed z dimensions.
    zM = self.dense_outM(zM)
    
    #left
    # Normalize.
    zL = self.z_normL(mfccsL[:, :, tf.newaxis, :])[:, :, 0, :]
    # Run an RNN over the latents.
    zL = self.rnnL(zL)
    # Bounce down to compressed z dimensions.
    zL = self.dense_outL(zL)
    
    #right
    # Normalize.
    zR = self.z_normR(mfccsR[:, :, tf.newaxis, :])[:, :, 0, :]
    # Run an RNN over the latents.
    zR = self.rnnR(zR)
    # Bounce down to compressed z dimensions.
    zR = self.dense_outR(zR)
    return zM, zL, zR


class F0Encoder(tfkl.Layer):
  """Mixin for F0 encoders."""

  def call(self, conditioning):
    return self.compute_f0(conditioning)

  def compute_f0(self, conditioning):
    """Takes in conditioning dictionary, returns fundamental frequency."""
    raise NotImplementedError

  def _compute_unit_midi(self, probs):
    """Computes the midi from a distribution over the unit interval."""
    # probs: [B, T, D]
    depth = int(probs.shape[-1])

    unit_midi_bins = tf.constant(
        1.0 * np.arange(depth).reshape((1, 1, -1)) / depth,
        dtype=tf.float32)  # [1, 1, D]

    f0_unit_midi = tf.reduce_sum(
        unit_midi_bins * probs, axis=-1, keepdims=True)  # [B, T, 1]
    return f0_unit_midi


@gin.register
class ResnetF0Encoder(F0Encoder):
  """Embeddings from resnet on spectrograms."""
  # adapted for stereo

  def __init__(self,
               size='large',
               f0_bins=384,
               spectral_fn=lambda x: spectral_ops.compute_mag(x, size=3072),
               name='resnet_f0_encoder'):
    super().__init__(name=name)
    self.f0_bins = f0_bins
    self.spectral_fn = spectral_fn

    # Layers.
    self.resnetM = nn.resnet(size=size)
    self.dense_outM = nn.dense(f0_bins)
    self.resnetL = nn.resnet(size=size)
    self.dense_outL = nn.dense(f0_bins)
    self.resnetR = nn.resnet(size=size)
    self.dense_outR = nn.dense(f0_bins)

  def compute_f0(self, conditioning):
    """Compute fundamental frequency."""
    magM = self.spectral_fn(conditioning['audioM'])
    magL = self.spectral_fn(conditioning['audioL'])
    magR = self.spectral_fn(conditioning['audioR'])
    magM = magM[:, :, :, tf.newaxis]
    magL = magL[:, :, :, tf.newaxis]
    magR = magR[:, :, :, tf.newaxis]
    xM = self.resnet(magM)
    xL = self.resnet(magL)
    xR = self.resnet(magR)

    # Collapse the frequency dimension
    x_shapeM = xM.shape.as_list()
    x_shapeL = xL.shape.as_list()
    x_shapeR = xR.shape.as_list()
    yM = tf.reshape(xM, [x_shapeM[0], x_shapeM[1], -1])
    yL = tf.reshape(xL, [x_shapeL[0], x_shapeL[1], -1])
    yR = tf.reshape(xR, [x_shapeR[0], x_shapeR[1], -1])
    # Project to f0_bins
    yM = self.dense_outM(y)
    yL = self.dense_outL(y)
    yR = self.dense_outR(y)

    # treat the NN output as probability over midi values.
    # probs = tf.nn.softmax(y)  # softmax leads to NaNs
    probsM = tf.nn.softplus(yM) + 1e-3
    probsL = tf.nn.softplus(yL) + 1e-3
    probsR = tf.nn.softplus(yR) + 1e-3
    probsM = probsM / tf.reduce_sum(probsM, axis=-1, keepdims=True)
    probsL = probsL / tf.reduce_sum(probsL, axis=-1, keepdims=True)
    probsR = probsR / tf.reduce_sum(probsR, axis=-1, keepdims=True)
    f0M = self._compute_unit_midi(probsM)
    f0L = self._compute_unit_midi(probsL)
    f0R = self._compute_unit_midi(probsR)

    # Make same time resolution as original CREPE f0.
    n_timestepsM = int(conditioning['f0_scaledM'].shape[1])
    n_timestepsL = int(conditioning['f0_scaledL'].shape[1])
    n_timestepsR = int(conditioning['f0_scaledR'].shape[1])
    f0M = resample(f0M, n_timestepsM)
    f0L = resample(f0L, n_timestepsL)
    f0R = resample(f0R, n_timestepsR)
    return f0M, f0L, f0R

def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)
        
def midi_to_hz(notes: Number) -> Number:
  """TF-compatible midi_to_hz function."""
  notes = tf_float32(notes)
  return 440.0 * (2.0**((notes - 69.0) / 12.0))
    
def resample(inputs: tf.Tensor,
             n_timesteps: int,
             method: Text = 'linear',
             add_endpoint: bool = True) -> tf.Tensor:
  """Interpolates a tensor from n_frames to n_timesteps.

  Args:
    inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
      [batch_size, n_frames], [batch_size, n_frames, channels], or
      [batch_size, n_frames, n_freq, channels].
    n_timesteps: Time resolution of the output signal.
    method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
      'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
      'window' uses overlapping windows (only for upsampling) which is smoother
      for amplitude envelopes with large frame sizes.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
      [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
      [batch_size, n_timesteps, n_freqs, channels].

  Raises:
    ValueError: If method is 'window' and input is 4-D.
    ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
      'window'.
  """
  inputs = tf_float32(inputs)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[tf.newaxis, :, tf.newaxis]
  elif is_2d:
    inputs = inputs[:, :, tf.newaxis]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = tf.compat.v1.image.resize(outputs,
                                        [n_timesteps, outputs.shape[2]],
                                        method=method,
                                        align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)
  elif method == 'linear':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BILINEAR)
  elif method == 'cubic':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BICUBIC)
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs


