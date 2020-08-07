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

import core
from training import nn
import gin
import tensorflow.compat.v2 as tf

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
class Decoder(tfkl.Layer):
  """Base class to implement any decoder.

  Users should override decode() to define the actual encoder structure.
  Hyper-parameters will be passed through the constructor.
  """
  # adapted for stereo

  def __init__(self,
               output_splitsM=(('amps', 1), ('harmonic_distribution', 120), ('noise_magnitudes', 65)),
               output_splitsL=(('amps', 1), ('harmonic_distribution', 120), ('noise_magnitudes', 65)),
               output_splitsR=(('amps', 1), ('harmonic_distribution', 120), ('noise_magnitudes', 65)),
               name=None):
    super().__init__(name=name)
    self.output_splitsM = output_splitsM
    self.output_splitsL = output_splitsL
    self.output_splitsR = output_splitsR
    self.n_outM = sum([v[1] for v in output_splitsM])
    self.n_outL = sum([v[1] for v in output_splitsL])
    self.n_outR = sum([v[1] for v in output_splitsR])

  def call(self, conditioning, mode):
    """Updates conditioning with dictionary of decoder outputs."""
    conditioning = core.copy_if_tf_function(conditioning)
    x = self.decode(conditioning, mode)
    outputsM = nn.split_to_dict(x, self.output_splitsM)
    outputsL = nn.split_to_dict(x, self.output_splitsL)
    outputsR = nn.split_to_dict(x, self.output_splitsR)

    if isinstance(outputsM, dict):
      conditioning.update(outputsM)
    else:
      raise ValueError('Decoder must output a dictionary of signals.')
    if isinstance(outputsL, dict):
      conditioning.update(outputsL)
    else:
      raise ValueError('Decoder must output a dictionary of signals.')
    if isinstance(outputsR, dict):
      conditioning.update(outputsR)
    else:
      raise ValueError('Decoder must output a dictionary of signals.')
    return conditioning

  def decode(self, conditioning):
    """Takes in conditioning dictionary, returns dictionary of signals."""
    raise NotImplementedError


@gin.register
class RnnFcDecoder(Decoder):
  """RNN and FC stacks for f0 and loudness."""
  # adapted for stereo
    
  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               input_keysM=('ld_scaledM', 'f0_scaledM', 'zM'),
               input_keysL=('ld_scaledL', 'f0_scaledL', 'zL'),
               input_keysR=('ld_scaledR', 'f0_scaledR', 'zR'),
               output_splitsM=(('amps', 1), ('harmonic_distribution', 120), ('noise_magnitudes', 65)),
               output_splitsL=(('amps', 1), ('harmonic_distribution', 120), ('noise_magnitudes', 65)),
               output_splitsR=(('amps', 1), ('harmonic_distribution', 120), ('noise_magnitudes', 65)),
               name=None):
    super().__init__(output_splitsM=output_splitsM, output_splitsL=output_splitsL, output_splitsR=output_splitsR, name=name)
    stackM = lambda: nn.fc_stack(ch, layers_per_stack)
    stackL = lambda: nn.fc_stack(ch, layers_per_stack)
    stackR = lambda: nn.fc_stack(ch, layers_per_stack)
    self.input_keysM = input_keysM
    self.input_keysL = input_keysL
    self.input_keysR = input_keysR

    # Layers.
    self.input_stacksM = [stackM() for k in self.input_keysM]
    self.input_stacksL = [stackL() for k in self.input_keysL]
    self.input_stacksR = [stackR() for k in self.input_keysR]
    self.rnnM = nn.rnn(rnn_channels, rnn_type)
    self.rnnL = nn.rnn(rnn_channels, rnn_type)
    self.rnnR = nn.rnn(rnn_channels, rnn_type)
    self.out_stackM = stackM()
    self.out_stackL = stackL()
    self.out_stackR = stackR()
    # Defines number of elements in output dictionaries
    self.dense_outM = nn.dense(self.n_outM)
    self.dense_outL = nn.dense(self.n_outL)
    self.dense_outR = nn.dense(self.n_outR)

    # Backwards compatability.
    self.f_stack = self.input_stacksM[0] if len(self.input_stacksM) >= 1 else None
    self.l_stack = self.input_stacksM[1] if len(self.input_stacksM) >= 2 else None
    self.z_stack = self.input_stacksM[2] if len(self.input_stacksM) >= 3 else None
    
    self.f_stack = self.input_stacksL[0] if len(self.input_stacksL) >= 1 else None
    self.l_stack = self.input_stacksL[1] if len(self.input_stacksL) >= 2 else None
    self.z_stack = self.input_stacksL[2] if len(self.input_stacksL) >= 3 else None
    
    self.f_stack = self.input_stacksR[0] if len(self.input_stacksR) >= 1 else None
    self.l_stack = self.input_stacksR[1] if len(self.input_stacksR) >= 2 else None
    self.z_stack = self.input_stacksR[2] if len(self.input_stacksR) >= 3 else None

  def decode(self, conditioning, mode):
    # Initial processing.
    inputsM = [conditioning[k] for k in self.input_keysM]
    inputsL = [conditioning[k] for k in self.input_keysL]
    inputsR = [conditioning[k] for k in self.input_keysR]
    inputsM = [stackM(x) for stackM, x in zip(self.input_stacksM, inputsM)]
    inputsL = [stackL(x) for stackL, x in zip(self.input_stacksL, inputsL)]
    inputsR = [stackR(x) for stackR, x in zip(self.input_stacksR, inputsR)]

    # Run an RNN over the latents.
    xM = tf.concat(inputsM, axis=-1)
    xM = self.rnnM(xM)
    xM = tf.concat(inputsM + [xM], axis=-1)
    
    xL = tf.concat(inputsL, axis=-1)
    xL = self.rnnL(xL)
    xL = tf.concat(inputsL + [xL], axis=-1)
    
    xR = tf.concat(inputsR, axis=-1)
    xR = self.rnnR(xR)
    xR = tf.concat(inputsR + [xR], axis=-1)

    # Final processing.
    xM = self.out_stackM(xM)
    xL = self.out_stackL(xL)
    xR = self.out_stackR(xR)
    dM = self.dense_outM(xM)
    dL = self.dense_outL(xL)
    dR = self.dense_outR(xR)
    
    if (mode == 'mono'):
      return dM
    if (mode == 'left'):
      return dL
    if (mode == 'right'):
      return dR
     


