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
"""Apache Beam pipeline for computing TFRecord dataset from audio files."""

from absl import logging
import apache_beam as beam
import librosa
import gin
import crepe
import numpy as np
from scipy.io.wavfile import read as read_audio
import scipy.signal as sps
import pydub
import tensorflow.compat.v2 as tf

F0_RANGE = 127.0  # MIDI
LD_RANGE = 120.0  # dB

_CREPE_SAMPLE_RATE = 16000
_CREPE_FRAME_SIZE = 1024

def _load_audio_as_array(audio_path: str,
                         sample_rate: int) -> np.array:
  """Load audio file at specified sample rate and return an array.

  When `sample_rate` > original SR of audio file, Pydub may miss samples when
  reading file defined in `audio_path`. Must manually zero-pad missing samples.

  Args:
    audio_path: path to audio file
    sample_rate: desired sample rate (can be different from original SR)

  Returns:
    audio: audio in np.float32
  """
  # adapted for stereo
  with tf.io.gfile.GFile(audio_path, 'rb') as f:
    # Load audio at original SR
    unused_sample_rate, wav = read_audio(f)
    expected_len = wav.shape[0]
    # Zero pad missing samples, if any
    # audio = pad_or_trim_to_expected_length(audio, expected_len)
    audio = pad_or_trim_to_expected_length(wav, expected_len)
  audio2 = np.copy(audio)
  audio2 = audio2 / (2**(8 * 2))
  audioM = np.squeeze(np.mean(audio2, axis=1))
  audioL = np.squeeze(audio2[:,0:1])
  audioR = np.squeeze(audio2[:,1:2])
  return audioL, audioR, audioM


def _load_audio(audio_path, sample_rate):
  """Load audio file."""
  # adapted for stereo
  logging.info("Loading '%s'.", audio_path)
  beam.metrics.Metrics.counter('prepare-tfrecord', 'load-audio').inc()
  # audio = _load_audio_as_array(audio_path, sample_rate)
  # return {'audio': audio}
  audioL, audioR, audioM = _load_audio_as_array(audio_path, sample_rate)
  audio_dict = {'audioL': audioL, 'audioR': audioR, 'audioM': audioM}
  return audio_dict


def add_loudness(ex, sample_rate, frame_rate, n_fft=6144):
  """Add loudness in dB."""
  # adapted for stereo
  beam.metrics.Metrics.counter('prepare-tfrecord', 'compute-loudness').inc()
  #STEREO
  audioM = ex['audioM']
  audioL = ex['audioL']
  audioR = ex['audioR']
  mean_loudness_dbM = compute_loudness(audioM, sample_rate, frame_rate, n_fft)
  mean_loudness_dbL = compute_loudness(audioL, sample_rate, frame_rate, n_fft)
  mean_loudness_dbR = compute_loudness(audioR, sample_rate, frame_rate, n_fft)
  ex = dict(ex)
  ex['loudness_dbM'] = mean_loudness_dbM.astype(np.float32)
  ex['loudness_dbL'] = mean_loudness_dbL.astype(np.float32)
  ex['loudness_dbR'] = mean_loudness_dbR.astype(np.float32)
  return ex
  
def _add_f0_estimate(ex, sample_rate, frame_rate):
  """Add fundamental frequency (f0) estimate using CREPE."""
  # adapted for stereo
  beam.metrics.Metrics.counter('prepare-tfrecord', 'estimate-f0').inc()
  #STEREO 
  audioM = ex['audioM']
  audioL = ex['audioL']
  audioR = ex['audioR']
  f0_hzM, f0_confidenceM = compute_f0(audioM, sample_rate, frame_rate)
  f0_hzL, f0_confidenceL = compute_f0(audioL, sample_rate, frame_rate)
  f0_hzR, f0_confidenceR = compute_f0(audioR, sample_rate, frame_rate)
  ex = dict(ex)
  ex.update({
      'f0_hzM': f0_hzM.astype(np.float32),
      'f0_confidenceM': f0_confidenceM.astype(np.float32),
      'f0_hzL': f0_hzL.astype(np.float32),
      'f0_confidenceL': f0_confidenceL.astype(np.float32),
      'f0_hzR': f0_hzR.astype(np.float32),
      'f0_confidenceR': f0_confidenceR.astype(np.float32)
  })
  return ex

    
def split_example(
    ex, sample_rate, frame_rate, window_secs, hop_secs):
  """Splits example into windows, padding final window if needed."""
  # adapted for stereo

  def get_windows(sequence, rate):
    window_size = int(window_secs * rate)
    hop_size = int(hop_secs * rate)
    n_windows = int(np.ceil((len(sequence) - window_size) / hop_size))  + 1
    n_samples_padded = (n_windows - 1) * hop_size + window_size
    n_padding = n_samples_padded - len(sequence)
    sequence = np.pad(sequence, (0, n_padding), mode='constant')
    for window_end in range(window_size, len(sequence) + 1, hop_size):
      yield sequence[window_end-window_size:window_end]
  
  for audioM, audioL, audioR, loudness_dbM, loudness_dbL, loudness_dbR, f0_hzM, f0_hzL, f0_hzR, f0_confidenceM, f0_confidenceL, f0_confidenceR in zip(
      get_windows(ex['audioM'], sample_rate),
      get_windows(ex['audioL'], sample_rate),
      get_windows(ex['audioR'], sample_rate),
      get_windows(ex['loudness_dbM'], frame_rate),
      get_windows(ex['loudness_dbL'], frame_rate),
      get_windows(ex['loudness_dbR'], frame_rate),
      get_windows(ex['f0_hzM'], frame_rate),
      get_windows(ex['f0_hzL'], frame_rate),
      get_windows(ex['f0_hzR'], frame_rate),
      get_windows(ex['f0_confidenceM'], frame_rate),
      get_windows(ex['f0_confidenceL'], frame_rate),
      get_windows(ex['f0_confidenceR'], frame_rate)):
    beam.metrics.Metrics.counter('prepare-tfrecord', 'split-example').inc()
    yield {
        'audioM': audioM,
        'audioL': audioL,
        'audioR': audioR,
        'loudness_dbM': loudness_dbM,
        'loudness_dbL': loudness_dbL,
        'loudness_dbR': loudness_dbR,
        'f0_hzM': f0_hzM,
        'f0_hzL': f0_hzL,
        'f0_hzR': f0_hzR,
        'f0_confidenceM': f0_confidenceM,
        'f0_confidenceL': f0_confidenceL,
        'f0_confidenceR': f0_confidenceR
    }


def float_dict_to_tfexample(float_dict):
  """Convert dictionary of float arrays to tf.train.Example proto."""
    
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              k: tf.train.Feature(float_list=tf.train.FloatList(value=v.tolist()))
              for k, v in float_dict.items()
          }
      ))


def prepare_tfrecord(
    input_audio_paths,
    output_tfrecord_path,
    num_shards=None,
    sample_rate=48000,
    frame_rate=250,
    window_secs=4,
    hop_secs=1,
    pipeline_options=''):
  """Prepares a TFRecord for use in training, evaluation, and prediction.

  Args:
    input_audio_paths: An iterable of paths to audio files to include in
      TFRecord.
    output_tfrecord_path: The prefix path to the output TFRecord. Shard numbers
      will be added to actual path(s).
    num_shards: The number of shards to use for the TFRecord. If None, this
      number will be determined automatically.
    sample_rate: The sample rate to use for the audio.
    frame_rate: The frame rate to use for f0 and loudness features.
      If set to None, these features will not be computed.
    window_secs: The size of the sliding window (in seconds) to use to
      split the audio and features. If 0, they will not be split.
    hop_secs: The number of seconds to hop when computing the sliding
      windows.
    pipeline_options: An iterable of command line arguments to be used as
      options for the Beam Pipeline.
  """
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      pipeline_options)
  with beam.Pipeline(options=pipeline_options) as pipeline:
    examples = (
        pipeline
        | beam.Create(input_audio_paths)
        | beam.Map(_load_audio, sample_rate))

    if frame_rate:
      examples = (
          examples
          | beam.Map(_add_f0_estimate, sample_rate, frame_rate)
          | beam.Map(add_loudness, sample_rate, frame_rate))

    if window_secs:
      examples |= beam.FlatMap(
          split_example, sample_rate, frame_rate, window_secs, hop_secs)

    _ = (
        examples
        | beam.Reshuffle()
        | beam.Map(float_dict_to_tfexample)
        | beam.io.tfrecordio.WriteToTFRecord(
            output_tfrecord_path,
            num_shards=num_shards,
            coder=beam.coders.ProtoCoder(tf.train.Example))
    )
    

def stft(audio, frame_size=6144, overlap=0.75, pad_end=True):
  """Differentiable stft in tensorflow, computed in batch."""
  audio = tf_float32(audio)
  assert frame_size * overlap % 2.0 == 0.0
  s = tf.signal.stft(
      signals=audio,
      frame_length=int(frame_size),
      frame_step=int(frame_size * (1.0 - overlap)),
      fft_length=int(frame_size),
      pad_end=pad_end)
  return s


def stft_np(audio, frame_size=6144, overlap=0.75, pad_end=True):
  """Non-differentiable stft using librosa, one example at a time."""
  assert frame_size * overlap % 2.0 == 0.0
  hop_size = int(frame_size * (1.0 - overlap))
  is_2d = (len(audio.shape) == 2)

  if pad_end:
    n_samples_initial = int(audio.shape[-1])
    n_frames = int(np.ceil(n_samples_initial / hop_size))
    n_samples_final = (n_frames - 1) * hop_size + frame_size
    pad = n_samples_final - n_samples_initial
    padding = ((0, 0), (0, pad)) if is_2d else ((0, pad),)
    audio = np.pad(audio, padding, 'constant')

  def stft_fn(y):
    return librosa.stft(y=y,
                        n_fft=int(frame_size),
                        hop_length=hop_size,
                        center=False).T

  s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
  return s
  
  
@gin.register
def compute_loudness(audio,
                     sample_rate=48000,
                     frame_rate=250,
                     n_fft=6144,
                     range_db=LD_RANGE,
                     ref_db=20.7,
                     use_tf=False):
  """Perceptual loudness in dB, relative to white noise, amplitude=1.

  Function is differentiable if use_tf=True.
  Args:
    audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
      [batch_size,].
    sample_rate: Audio sample rate in Hz.
    frame_rate: Rate of loudness frames in Hz.
    n_fft: Fft window size.
    range_db: Sets the dynamic range of loudness in decibles. The minimum
      loudness (per a frequency bin) corresponds to -range_db.
    ref_db: Sets the reference maximum perceptual loudness as given by
      (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
      corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
      slight dependence on fft_size due to different granularity of perceptual
      weighting.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
  """
  if sample_rate % frame_rate != 0:
    raise ValueError(
        'frame_rate: {} must evenly divide sample_rate: {}.'
        'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
        .format(frame_rate, sample_rate))

  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  # Make inputs tensors for tensorflow.
  audio = tf_float32(audio) if use_tf else audio

  # Temporarily a batch dimension for single examples.
  is_1d = (len(audio.shape) == 1)
  audio = audio[lib.newaxis, :] if is_1d else audio

  # Take STFT.
  hop_size = sample_rate // frame_rate
  overlap = 1 - hop_size / n_fft
  stft_fn = stft if use_tf else stft_np
  s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=True)

  # Compute power
  amplitude = lib.abs(s)
  log10 = (lambda x: tf.math.log(x) / tf.math.log(10.0)) if use_tf else np.log10
  amin = 1e-20  # Avoid log(0) instabilities.
  power_db = log10(lib.maximum(amin, amplitude))
  power_db *= 20.0

  # Perceptual weighting.
  frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
  a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]
  loudness = power_db + a_weighting

  # Set dynamic range.
  loudness -= ref_db
  loudness = lib.maximum(loudness, -range_db)
  mean = tf.reduce_mean if use_tf else np.mean

  # Average over frequency bins.
  loudness = mean(loudness, axis=-1)

  # Remove temporary batch dimension.
  loudness = loudness[0] if is_1d else loudness

  # Compute expected length of loudness vector
  n_secs = audio.shape[-1] / float(
      sample_rate)  # `n_secs` can have milliseconds
  expected_len = int(n_secs * frame_rate)

  # Pad with `-range_db` noise floor or trim vector
  loudness = pad_or_trim_to_expected_length_mono(
      loudness, expected_len, -range_db, use_tf=use_tf)
  return loudness


@gin.register
def compute_f0(audio, sample_rate, frame_rate, viterbi=True):
  """Fundamental frequency (f0) estimate using CREPE.

  This function is non-differentiable and takes input as a numpy array.
  Args:
    audio: Numpy ndarray of single audio example. Shape [audio_length,].
    sample_rate: Sample rate in Hz.
    frame_rate: Rate of f0 frames in Hz.
    viterbi: Use Viterbi decoding to estimate f0.

  Returns:
    f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
    f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
  """

  n_secs = len(audio) / float(sample_rate)  # `n_secs` can have milliseconds
  crepe_step_size = 1000 / frame_rate  # milliseconds
  expected_len = int(n_secs * frame_rate)
  audio = np.asarray(audio)
  # sample_rate = _CREPE_SAMPLE_RATE

  # Compute f0 with crepe.
  _, f0_hz, f0_confidence, _ = crepe.predict(
      audio,
      sr=sample_rate,
      viterbi=viterbi,
      step_size=crepe_step_size,
      center=False,
      verbose=0)

  # Postprocessing on f0_hz
  f0_hz = pad_or_trim_to_expected_length_mono(f0_hz, expected_len, 0)  # pad with 0
  f0_hz = f0_hz.astype(np.float32)

  # Postprocessing on f0_confidence
  f0_confidence = pad_or_trim_to_expected_length_mono(f0_confidence, expected_len, 1)
  # f0_confidence = pad_or_trim_to_expected_length(f0_confidence, expected_len, 2)
  f0_confidence = np.nan_to_num(f0_confidence)   # Set nans to 0 in confidence
  f0_confidence = f0_confidence.astype(np.float32)
  return f0_hz, f0_confidence


def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20,
                                   use_tf=False):
  """Make vector equal to the expected length.

  Feature extraction functions like `compute_loudness()` or `compute_f0` produce
  feature vectors that vary in length depending on factors such as `sample_rate`
  or `hop_size`. This function corrects vectors to the expected length, warning
  the user if the difference between the vector and expected length was
  unusually high to begin with.

  Args:
    vector: Numpy 1D ndarray. Shape [vector_length,]
    expected_len: Expected length of vector.
    pad_value: Value to pad at end of vector.
    len_tolerance: Tolerance of difference between original and desired vector
      length.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    vector: Vector with corrected length.

  Raises:
    ValueError: if `len(vector)` is different from `expected_len` beyond
    `len_tolerance` to begin with.
  """
  expected_len = int(expected_len)
  vector_len = int(vector.shape[0])

  if abs(vector_len - expected_len) > len_tolerance:
    # Ensure vector was close to expected length to begin with
    raise ValueError('Vector length: {} differs from expected length: {} '
                     'beyond tolerance of : {}'.format(vector_len,
                                                       expected_len,
                                                       len_tolerance))
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  # Pad missing samples
  if vector_len < expected_len:
    n_padding = expected_len - vector_len
    vector = lib.pad(
        vector, ((0, n_padding), (0, 0)),
        mode='constant',
        constant_values=pad_value)
  # Trim samples
  elif vector_len > expected_len:
    vector = vector[:expected_len, ...]

  return vector
  

def pad_or_trim_to_expected_length_mono(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20,
                                   use_tf=False):
  """Make vector equal to the expected length.

  Feature extraction functions like `compute_loudness()` or `compute_f0` produce
  feature vectors that vary in length depending on factors such as `sample_rate`
  or `hop_size`. This function corrects vectors to the expected length, warning
  the user if the difference between the vector and expected length was
  unusually high to begin with.

  Args:
    vector: Numpy 1D ndarray. Shape [vector_length,]
    expected_len: Expected length of vector.
    pad_value: Value to pad at end of vector.
    len_tolerance: Tolerance of difference between original and desired vector
      length.
    use_tf: Make function differentiable by using tensorflow.

  Returns:
    vector: Vector with corrected length.

  Raises:
    ValueError: if `len(vector)` is different from `expected_len` beyond
    `len_tolerance` to begin with.
  """
  expected_len = int(expected_len)
  vector_len = int(vector.shape[-1])

  if abs(vector_len - expected_len) > len_tolerance:
    # Ensure vector was close to expected length to begin with
    raise ValueError('Vector length: {} differs from expected length: {} '
                     'beyond tolerance of : {}'.format(vector_len,
                                                       expected_len,
                                                       len_tolerance))
  # Pick tensorflow or numpy.
  lib = tf if use_tf else np

  is_1d = (len(vector.shape) == 1)
  vector = vector[lib.newaxis, :] if is_1d else vector

  # Pad missing samples
  if vector_len < expected_len:
    n_padding = expected_len - vector_len
    vector = lib.pad(
        vector, ((0, 0), (0, n_padding)),
        mode='constant',
        constant_values=pad_value)
  # Trim samples
  elif vector_len > expected_len:
    vector = vector[..., :expected_len]

  # Remove temporary batch dimension.
  vector = vector[0] if is_1d else vector
  return vector


def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)