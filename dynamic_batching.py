# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

batcher_ops = tf.load_op_library('./batcher.so')

nest = tf.contrib.framework.nest


class _Batcher(object):
  """A thin layer around the Batcher TensorFlow operations.

  It shares some of the interface with queues (close(), name) to be able to use
  it correctly as the input to a QueueRunner.
  """

  def __init__(self, minimum_batch_size, maximum_batch_size, timeout_ms):
    self._handle = batcher_ops.batcher(minimum_batch_size, maximum_batch_size,
                                       timeout_ms or -1)

  @property
  def name(self):
    return 'batcher'

  def get_inputs(self, input_dtypes):
    return batcher_ops.batcher_get_inputs(self._handle, input_dtypes)

  def set_outputs(self, flat_result, computation_id):
    return batcher_ops.batcher_set_outputs(self._handle, flat_result,
                                           computation_id)

  def compute(self, flat_args, output_dtypes):
    return batcher_ops.batcher_compute(self._handle, flat_args, output_dtypes)

  def close(self, cancel_pending_enqueues=False, name=None):
    del cancel_pending_enqueues
    return batcher_ops.batcher_close(self._handle, name=name)


def batch_fn(f):
  """See `batch_fn_with_options` for details."""
  return batch_fn_with_options()(f)


def batch_fn_with_options(minimum_batch_size=1, maximum_batch_size=1024,
                          timeout_ms=100):
  """Python decorator that automatically batches computations.

  When the decorated function is called, it creates an operation that adds the
  inputs to a queue, waits until the computation is done, and returns the
  tensors. The inputs must be nests (see `tf.contrib.framework.nest`) and the
  first dimension of each tensor in the nest must have size 1.

  It adds a QueueRunner that asynchronously keeps fetching batches of data,
  computes the results and pushes the results back to the caller.

  Example usage:

    @dynamic_batching.batch_fn_with_options(
        minimum_batch_size=10, timeout_ms=100)
    def fn(a, b):
      return a + b

    output0 = fn(tf.constant([1]), tf.constant([2]))  # Will be batched with the
                                                      # next call.
    output1 = fn(tf.constant([3]), tf.constant([4]))

  Note, gradients are currently not supported.
  Note, if minimum_batch_size == maximum_batch_size and timeout_ms=None, then
  the batch size of input arguments will be set statically. Otherwise, it will
  be None.

  Args:
    minimum_batch_size: The minimum batch size before processing starts.
    maximum_batch_size: The maximum batch size.
    timeout_ms: Milliseconds after a batch of samples is requested before it is
      processed, even if the batch size is smaller than `minimum_batch_size`. If
      None, there is no timeout.

  Returns:
    The decorator.
  """

  def decorator(f):
    """Decorator."""
    batcher = [None]
    batched_output = [None]

    @functools.wraps(f)
    def wrapper(*args):
      """Wrapper."""

      flat_args = [tf.convert_to_tensor(arg) for arg in nest.flatten(args)]

      if batcher[0] is None:
        # Remove control dependencies which is necessary when created in loops,
        # etc.
        with tf.control_dependencies(None):
          input_dtypes = [t.dtype for t in flat_args]
          batcher[0] = _Batcher(minimum_batch_size, maximum_batch_size,
                                timeout_ms)

          # Compute in batches using a queue runner.

          if minimum_batch_size == maximum_batch_size and timeout_ms is None:
            batch_size = minimum_batch_size
          else:
            batch_size = None

          # Dequeue batched input.
          inputs, computation_id = batcher[0].get_inputs(input_dtypes)
          nest.map_structure(
              lambda i, a: i.set_shape([batch_size] + a.shape.as_list()[1:]),
              inputs, flat_args)

          # Compute result.
          result = f(*nest.pack_sequence_as(args, inputs))
          batched_output[0] = result
          flat_result = nest.flatten(result)

          # Insert results back into batcher.
          set_op = batcher[0].set_outputs(flat_result, computation_id)

          tf.train.add_queue_runner(tf.train.QueueRunner(batcher[0], [set_op]))

      # Insert inputs into input queue.
      flat_result = batcher[0].compute(
          flat_args,
          [t.dtype for t in nest.flatten(batched_output[0])])

      # Restore structure and shapes.
      result = nest.pack_sequence_as(batched_output[0], flat_result)
      static_batch_size = nest.flatten(args)[0].shape[0]

      nest.map_structure(
          lambda t, b: t.set_shape([static_batch_size] + b.shape[1:].as_list()),
          result, batched_output[0])
      return result

    return wrapper

  return decorator
