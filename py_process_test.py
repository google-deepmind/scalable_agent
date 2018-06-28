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

"""Tests py_process.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import time

import numpy as np
import py_process
import tensorflow as tf

from six.moves import range


class PyProcessTest(tf.test.TestCase):

  def test_small(self):

    class Example(object):

      def __init__(self, a):
        self._a = a

      def inc(self):
        self._a += 1

      def compute(self, b):
        return np.array(self._a + b, dtype=np.int32)

      @staticmethod
      def _tensor_specs(method_name, unused_args, unused_constructor_kwargs):
        if method_name == 'compute':
          return tf.contrib.framework.TensorSpec([], tf.int32)
        elif method_name == 'inc':
          return ()

    with tf.Graph().as_default():
      p = py_process.PyProcess(Example, 1)
      inc = p.proxy.inc()
      compute = p.proxy.compute(2)

      with tf.train.SingularMonitoredSession(
          hooks=[py_process.PyProcessHook()]) as session:
        self.assertTrue(isinstance(inc, tf.Operation))
        session.run(inc)

        self.assertEqual([], compute.shape)
        self.assertEqual(4, session.run(compute))

  def test_threading(self):

    class Example(object):

      def __init__(self):
        pass

      def wait(self):
        time.sleep(.2)
        return None

      @staticmethod
      def _tensor_specs(method_name, unused_args, unused_constructor_kwargs):
        if method_name == 'wait':
          return tf.contrib.framework.TensorSpec([], tf.int32)

    with tf.Graph().as_default():
      p = py_process.PyProcess(Example)
      wait = p.proxy.wait()

      hook = py_process.PyProcessHook()
      with tf.train.SingularMonitoredSession(hooks=[hook]) as session:

        def run():
          with self.assertRaises(tf.errors.OutOfRangeError):
            session.run(wait)

        t = self.checkedThread(target=run)
        t.start()
        time.sleep(.1)
      t.join()

  def test_args(self):

    class Example(object):

      def __init__(self, dim0):
        self._dim0 = dim0

      def compute(self, dim1):
        return np.zeros([self._dim0, dim1], dtype=np.int32)

      @staticmethod
      def _tensor_specs(method_name, kwargs, constructor_kwargs):
        dim0 = constructor_kwargs['dim0']
        dim1 = kwargs['dim1']
        if method_name == 'compute':
          return tf.contrib.framework.TensorSpec([dim0, dim1], tf.int32)

    with tf.Graph().as_default():
      p = py_process.PyProcess(Example, 1)
      result = p.proxy.compute(2)

      with tf.train.SingularMonitoredSession(
          hooks=[py_process.PyProcessHook()]) as session:
        self.assertEqual([1, 2], result.shape)
        self.assertAllEqual([[0, 0]], session.run(result))

  def test_error_handling_constructor(self):

    class Example(object):

      def __init__(self):
        raise ValueError('foo')

      def something(self):
        pass

      @staticmethod
      def _tensor_specs(method_name, unused_kwargs, unused_constructor_kwargs):
        if method_name == 'something':
          return ()

    with tf.Graph().as_default():
      py_process.PyProcess(Example, 1)

      with self.assertRaisesRegexp(Exception, 'foo'):
        with tf.train.SingularMonitoredSession(
            hooks=[py_process.PyProcessHook()]):
          pass

  def test_error_handling_method(self):

    class Example(object):

      def __init__(self):
        pass

      def something(self):
        raise ValueError('foo')

      @staticmethod
      def _tensor_specs(method_name, unused_kwargs, unused_constructor_kwargs):
        if method_name == 'something':
          return ()

    with tf.Graph().as_default():
      p = py_process.PyProcess(Example, 1)
      result = p.proxy.something()

      with tf.train.SingularMonitoredSession(
          hooks=[py_process.PyProcessHook()]) as session:
        with self.assertRaisesRegexp(Exception, 'foo'):
          session.run(result)

  def test_close(self):
    with tempfile.NamedTemporaryFile() as tmp:
      class Example(object):

        def __init__(self, filename):
          self._filename = filename

        def close(self):
          with tf.gfile.Open(self._filename, 'w') as f:
            f.write('was_closed')

      with tf.Graph().as_default():
        py_process.PyProcess(Example, tmp.name)

        with tf.train.SingularMonitoredSession(
            hooks=[py_process.PyProcessHook()]):
          pass

      self.assertEqual('was_closed', tmp.read())

  def test_close_on_error(self):
    with tempfile.NamedTemporaryFile() as tmp:

      class Example(object):

        def __init__(self, filename):
          self._filename = filename

        def something(self):
          raise ValueError('foo')

        def close(self):
          with tf.gfile.Open(self._filename, 'w') as f:
            f.write('was_closed')

        @staticmethod
        def _tensor_specs(method_name, unused_kwargs,
                          unused_constructor_kwargs):
          if method_name == 'something':
            return ()

      with tf.Graph().as_default():
        p = py_process.PyProcess(Example, tmp.name)
        result = p.proxy.something()

        with tf.train.SingularMonitoredSession(
            hooks=[py_process.PyProcessHook()]) as session:
          with self.assertRaisesRegexp(Exception, 'foo'):
            session.run(result)

      self.assertEqual('was_closed', tmp.read())


class PyProcessBenchmarks(tf.test.Benchmark):

  class Example(object):

    def __init__(self):
      self._result = np.random.randint(0, 256, (72, 96, 3), np.uint8)

    def compute(self, unused_a):
      return self._result

    @staticmethod
    def _tensor_specs(method_name, unused_args, unused_constructor_kwargs):
      if method_name == 'compute':
        return tf.contrib.framework.TensorSpec([72, 96, 3], tf.uint8)

  def benchmark_one(self):
    with tf.Graph().as_default():
      p = py_process.PyProcess(PyProcessBenchmarks.Example)
      compute = p.proxy.compute(2)

      with tf.train.SingularMonitoredSession(
          hooks=[py_process.PyProcessHook()]) as session:

        self.run_op_benchmark(
            name='process_one',
            sess=session,
            op_or_tensor=compute,
            burn_iters=10,
            min_iters=5000)

  def benchmark_many(self):
    with tf.Graph().as_default():
      ps = [
          py_process.PyProcess(PyProcessBenchmarks.Example) for _ in range(200)
      ]
      compute_ops = [p.proxy.compute(2) for p in ps]
      compute = tf.group(*compute_ops)

      with tf.train.SingularMonitoredSession(
          hooks=[py_process.PyProcessHook()]) as session:

        self.run_op_benchmark(
            name='process_many',
            sess=session,
            op_or_tensor=compute,
            burn_iters=10,
            min_iters=500)


if __name__ == '__main__':
  tf.test.main()
