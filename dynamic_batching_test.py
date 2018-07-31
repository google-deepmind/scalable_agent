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

"""Tests dynamic_batching.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
from multiprocessing import pool
import time

import dynamic_batching

import tensorflow as tf

from six.moves import range


_SLEEP_TIME = 1.0


class DynamicBatchingTest(tf.test.TestCase):

  def test_one(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn
      def f(a, b):
        batch_size = tf.shape(a)[0]
        return a + b, tf.tile([batch_size], [batch_size])

      output = f(tf.constant([[1, 3]]), tf.constant([2]))

      tf.train.start_queue_runners()

      result, batch_size = session.run(output)

      self.assertAllEqual([[3, 5]], result)
      self.assertAllEqual([1], batch_size)

  def test_two(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn
      def f(a, b):
        batch_size = tf.shape(a)[0]
        return a + b, tf.tile([batch_size], [batch_size])

      output0 = f(tf.constant([1]), tf.constant([2]))
      output1 = f(tf.constant([2]), tf.constant([3]))

      tp = pool.ThreadPool(2)
      f0 = tp.apply_async(session.run, [output0])
      f1 = tp.apply_async(session.run, [output1])

      # Make sure both inputs are in the batcher before starting it.
      time.sleep(_SLEEP_TIME)

      tf.train.start_queue_runners()

      result0, batch_size0 = f0.get()
      result1, batch_size1 = f1.get()

      self.assertAllEqual([3], result0)
      self.assertAllEqual([2], batch_size0)
      self.assertAllEqual([5], result1)
      self.assertAllEqual([2], batch_size1)

  def test_many_small(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn
      def f(a, b):
        return a + b

      outputs = []
      for i in range(200):
        outputs.append(f(tf.fill([1, 5], i), tf.fill([1, 5], i)))

      tf.train.start_queue_runners()

      tp = pool.ThreadPool(10)
      futures = []
      for output in outputs:
        futures.append(tp.apply_async(session.run, [output]))

      for i, future in enumerate(futures):
        result = future.get()
        self.assertAllEqual([[i * 2] * 5], result)

  def test_input_batch_size_should_be_one(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn
      def f(a):
        return a

      output = f(tf.constant([1, 2]))

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)

      with self.assertRaises(tf.errors.CancelledError):
        session.run(output)

      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'requires batch size 1'):
        coord.join()

  def test_run_after_error_should_be_cancelled(self):
    with self.test_session() as session:

      @dynamic_batching.batch_fn
      def f(a):
        return a

      output = f(tf.constant([1, 2]))

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)

      with self.assertRaises(tf.errors.CancelledError):
        session.run(output)

      with self.assertRaises(tf.errors.CancelledError):
        session.run(output)

  def test_input_shapes_should_be_equal(self):
    with self.test_session() as session:

      @dynamic_batching.batch_fn
      def f(a, b):
        return a + b

      output0 = f(tf.constant([1]), tf.constant([2]))
      output1 = f(tf.constant([[2]]), tf.constant([3]))

      tp = pool.ThreadPool(2)
      f0 = tp.apply_async(session.run, [output0])
      f1 = tp.apply_async(session.run, [output1])

      time.sleep(_SLEEP_TIME)

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)

      with self.assertRaises(tf.errors.CancelledError):
        f0.get()
        f1.get()

      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'Shapes of inputs much be equal'):
        coord.join()

  def test_output_must_have_batch_dimension(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn
      def f(_):
        return tf.constant(1)

      output = f(tf.constant([1]))

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)

      with self.assertRaises(tf.errors.CancelledError):
        session.run(output)

      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'Output shape must have a batch dimension'):
        coord.join()

  def test_output_must_have_same_batch_dimension_size_as_input(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn
      def f(_):
        return tf.constant([1, 2, 3, 4])

      output = f(tf.constant([1]))

      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(coord=coord)

      with self.assertRaises(tf.errors.CancelledError):
        session.run(output)

      with self.assertRaisesRegexp(
          tf.errors.InvalidArgumentError,
          'Output shape must have the same batch dimension as the input batch '
          'size. Expected: 1 Observed: 4'):
        coord.join()

  def test_get_inputs_cancelled(self):
    with tf.Graph().as_default():

      @dynamic_batching.batch_fn
      def f(a):
        return a

      f(tf.constant([1]))

      # Intentionally using tf.Session() instead of self.test_session() to have
      # control over closing the session. test_session() is a cached session.
      with tf.Session():
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord)
        # Sleep to make sure the queue runner has started the first run call.
        time.sleep(_SLEEP_TIME)

      # Session closed.
      with self.assertRaisesRegexp(tf.errors.CancelledError,
                                   'GetInputs operation was cancelled'):
        coord.join()

  def test_batcher_closed(self):
    with tf.Graph().as_default():
      @dynamic_batching.batch_fn
      def f(a):
        return a

      f(tf.constant([1]))

      # Intentionally using tf.Session() instead of self.test_session() to have
      # control over closing the session. test_session() is a cached session.
      with tf.Session():
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord)
        time.sleep(_SLEEP_TIME)
        coord.request_stop()  # Calls close operation.
        coord.join()
      # Session closed.

  def test_minimum_batch_size(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn_with_options(
          minimum_batch_size=2, timeout_ms=1000)
      def f(a, b):
        batch_size = tf.shape(a)[0]
        return a + b, tf.tile([batch_size], [batch_size])

      output = f(tf.constant([[1, 3]]), tf.constant([2]))

      tf.train.start_queue_runners()

      start = datetime.datetime.now()
      session.run(output)
      duration = datetime.datetime.now() - start

      # There should have been a timeout here because only one sample was added
      # and the minimum batch size is 2.
      self.assertLessEqual(.9, duration.total_seconds())
      self.assertGreaterEqual(1.5, duration.total_seconds())

      outputs = [
          f(tf.constant([[1, 3]]), tf.constant([2])),
          f(tf.constant([[1, 3]]), tf.constant([2]))
      ]

      start = datetime.datetime.now()
      (_, batch_size), _ = session.run(outputs)
      duration = datetime.datetime.now() - start

      # The outputs should be executed immediately because two samples are
      # added.
      self.assertGreaterEqual(.5, duration.total_seconds())
      self.assertEqual(2, batch_size)

  def test_maximum_batch_size(self):
    with self.test_session() as session:
      @dynamic_batching.batch_fn_with_options(maximum_batch_size=2)
      def f(a, b):
        batch_size = tf.shape(a)[0]
        return a + b, tf.tile([batch_size], [batch_size])

      outputs = [
          f(tf.constant([1]), tf.constant([2])),
          f(tf.constant([1]), tf.constant([2])),
          f(tf.constant([1]), tf.constant([2])),
          f(tf.constant([1]), tf.constant([2])),
          f(tf.constant([1]), tf.constant([2])),
      ]

      tf.train.start_queue_runners()

      results = session.run(outputs)

      for value, batch_size in results:
        self.assertEqual(3, value)
        self.assertGreaterEqual(2, batch_size)

  def test_static_shape(self):
    assertions_triggered = [0]

    @dynamic_batching.batch_fn_with_options(minimum_batch_size=1,
                                            maximum_batch_size=2)
    def f0(a):
      self.assertEqual(None, a.shape[0].value)
      assertions_triggered[0] += 1
      return a

    @dynamic_batching.batch_fn_with_options(minimum_batch_size=2,
                                            maximum_batch_size=2)
    def f1(a):
      # Even though minimum_batch_size and maximum_batch_size are equal, the
      # timeout can cause a batch with less than mininum_batch_size.
      self.assertEqual(None, a.shape[0].value)
      assertions_triggered[0] += 1
      return a

    @dynamic_batching.batch_fn_with_options(minimum_batch_size=2,
                                            maximum_batch_size=2,
                                            timeout_ms=None)
    def f2(a):
      # When timeout is disabled and minimum/maximum batch size are equal, the
      # shape is statically known.
      self.assertEqual(2, a.shape[0].value)
      assertions_triggered[0] += 1
      return a

    f0(tf.constant([1]))
    f1(tf.constant([1]))
    f2(tf.constant([1]))
    self.assertEqual(3, assertions_triggered[0])

  def test_out_of_order_execution1(self):
    with self.test_session() as session:
      batcher = dynamic_batching._Batcher(minimum_batch_size=1,
                                          maximum_batch_size=1,
                                          timeout_ms=None)

      tp = pool.ThreadPool(10)
      r0 = tp.apply_async(session.run, batcher.compute([[1]], [tf.int32]))
      (input0,), computation_id0 = session.run(batcher.get_inputs([tf.int32]))
      r1 = tp.apply_async(session.run, batcher.compute([[2]], [tf.int32]))
      (input1,), computation_id1 = session.run(batcher.get_inputs([tf.int32]))

      self.assertAllEqual([1], input0)
      self.assertAllEqual([2], input1)

      session.run(batcher.set_outputs([input0 + 42], computation_id0))
      session.run(batcher.set_outputs([input1 + 42], computation_id1))

      self.assertAllEqual([43], r0.get())
      self.assertAllEqual([44], r1.get())

  def test_out_of_order_execution2(self):
    with self.test_session() as session:
      batcher = dynamic_batching._Batcher(minimum_batch_size=1,
                                          maximum_batch_size=1,
                                          timeout_ms=None)

      tp = pool.ThreadPool(10)
      r0 = tp.apply_async(session.run, batcher.compute([[1]], [tf.int32]))
      (input0,), computation_id0 = session.run(batcher.get_inputs([tf.int32]))
      r1 = tp.apply_async(session.run, batcher.compute([[2]], [tf.int32]))
      (input1,), computation_id1 = session.run(batcher.get_inputs([tf.int32]))

      self.assertAllEqual([1], input0)
      self.assertAllEqual([2], input1)

      # These two runs are switched from testOutOfOrderExecution1.
      session.run(batcher.set_outputs([input1 + 42], computation_id1))
      session.run(batcher.set_outputs([input0 + 42], computation_id0))

      self.assertAllEqual([43], r0.get())
      self.assertAllEqual([44], r1.get())

  def test_invalid_computation_id(self):
    with self.test_session() as session:
      batcher = dynamic_batching._Batcher(minimum_batch_size=1,
                                          maximum_batch_size=1,
                                          timeout_ms=None)

      tp = pool.ThreadPool(10)
      tp.apply_async(session.run, batcher.compute([[1]], [tf.int32]))
      (input0,), _ = session.run(batcher.get_inputs([tf.int32]))

      self.assertAllEqual([1], input0)

      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   'Invalid computation id'):
        session.run(batcher.set_outputs([input0], 42))

  def test_op_shape(self):
    with self.test_session():
      batcher = dynamic_batching._Batcher(minimum_batch_size=1,
                                          maximum_batch_size=1,
                                          timeout_ms=None)

      _, computation_id = batcher.get_inputs([tf.int32])

      self.assertEqual([], computation_id.shape)


class DynamicBatchingBenchmarks(tf.test.Benchmark):

  def benchmark_batching_small(self):
    with tf.Session() as session:
      @dynamic_batching.batch_fn
      def f(a, b):
        return a + b

      outputs = []
      for _ in range(1000):
        outputs.append(f(tf.ones([1, 10]), tf.ones([1, 10])))
      op_to_benchmark = tf.group(*outputs)

      tf.train.start_queue_runners()

      self.run_op_benchmark(
          name='batching_many_small',
          sess=session,
          op_or_tensor=op_to_benchmark,
          burn_iters=10,
          min_iters=50)

  def benchmark_batching_large(self):
    with tf.Session() as session:
      @dynamic_batching.batch_fn
      def f(a, b):
        return a + b

      outputs = []
      for _ in range(1000):
        outputs.append(f(tf.ones([1, 100000]), tf.ones([1, 100000])))
      op_to_benchmark = tf.group(*outputs)

      tf.train.start_queue_runners()

      self.run_op_benchmark(
          name='batching_many_large',
          sess=session,
          op_or_tensor=op_to_benchmark,
          burn_iters=10,
          min_iters=50)


if __name__ == '__main__':
  tf.test.main()
