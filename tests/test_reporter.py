import contextlib
import tempfile
import threading
import time
import unittest

import numpy
import torch
from torch.nn import functional

import pytorch_trainer
from pytorch_trainer import testing
from pytorch_trainer.testing import backend


class TestReporter(unittest.TestCase):

    def test_empty_reporter(self):
        reporter = pytorch_trainer.Reporter()
        self.assertEqual(reporter.observation, {})

    def test_enter_exit(self):
        reporter1 = pytorch_trainer.Reporter()
        reporter2 = pytorch_trainer.Reporter()
        with reporter1:
            self.assertIs(pytorch_trainer.get_current_reporter(), reporter1)
            with reporter2:
                self.assertIs(pytorch_trainer.get_current_reporter(), reporter2)
            self.assertIs(pytorch_trainer.get_current_reporter(), reporter1)

    def test_enter_exit_threadsafe(self):
        # This test ensures reporter.__enter__ correctly stores the reporter
        # in the thread-local storage.

        def thread_func(reporter, record):
            with reporter:
                # Sleep for a tiny moment to cause an overlap of the context
                # managers.
                time.sleep(0.01)
                record.append(pytorch_trainer.get_current_reporter())

        record1 = []  # The current repoter in each thread is stored here.
        record2 = []
        reporter1 = pytorch_trainer.Reporter()
        reporter2 = pytorch_trainer.Reporter()
        thread1 = threading.Thread(
            target=thread_func,
            args=(reporter1, record1))
        thread2 = threading.Thread(
            target=thread_func,
            args=(reporter2, record2))
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        self.assertIs(record1[0], reporter1)
        self.assertIs(record2[0], reporter2)

    def test_scope(self):
        reporter1 = pytorch_trainer.Reporter()
        reporter2 = pytorch_trainer.Reporter()
        with reporter1:
            observation = {}
            with reporter2.scope(observation):
                self.assertIs(pytorch_trainer.get_current_reporter(), reporter2)
                self.assertIs(reporter2.observation, observation)
            self.assertIs(pytorch_trainer.get_current_reporter(), reporter1)
            self.assertIsNot(reporter2.observation, observation)

    def test_add_observer(self):
        reporter = pytorch_trainer.Reporter()
        observer = object()
        reporter.add_observer('o', observer)

        reporter.report({'x': 1}, observer)

        observation = reporter.observation
        self.assertIn('o/x', observation)
        self.assertEqual(observation['o/x'], 1)
        self.assertNotIn('x', observation)

    def test_add_observers(self):
        reporter = pytorch_trainer.Reporter()
        observer1 = object()
        reporter.add_observer('o1', observer1)
        observer2 = object()
        reporter.add_observer('o2', observer2)

        reporter.report({'x': 1}, observer1)
        reporter.report({'y': 2}, observer2)

        observation = reporter.observation
        self.assertIn('o1/x', observation)
        self.assertEqual(observation['o1/x'], 1)
        self.assertIn('o2/y', observation)
        self.assertEqual(observation['o2/y'], 2)
        self.assertNotIn('x', observation)
        self.assertNotIn('y', observation)
        self.assertNotIn('o1/y', observation)
        self.assertNotIn('o2/x', observation)

    def test_report_without_observer(self):
        reporter = pytorch_trainer.Reporter()
        reporter.report({'x': 1})

        observation = reporter.observation
        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)


class TestNoKeepingGraphOnReportFlag(unittest.TestCase):

    def test_keep_graph_default(self):
        x = torch.from_numpy(numpy.array([1], numpy.float32)).requires_grad_(True)
        y = functional.sigmoid(x)
        reporter = pytorch_trainer.Reporter()
        reporter.report({'y': y})
        self.assertFalse(reporter.observation['y'].requires_grad)


class TestReport(unittest.TestCase):

    def test_report_without_reporter(self):
        observer = object()
        pytorch_trainer.report({'x': 1}, observer)

    def test_report(self):
        reporter = pytorch_trainer.Reporter()
        with reporter:
            pytorch_trainer.report({'x': 1})
        observation = reporter.observation
        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)

    def test_report_with_observer(self):
        reporter = pytorch_trainer.Reporter()
        observer = object()
        reporter.add_observer('o', observer)
        with reporter:
            pytorch_trainer.report({'x': 1}, observer)
        observation = reporter.observation
        self.assertIn('o/x', observation)
        self.assertEqual(observation['o/x'], 1)

    def test_report_with_unregistered_observer(self):
        reporter = pytorch_trainer.Reporter()
        observer = object()
        with reporter:
            with self.assertRaises(KeyError):
                pytorch_trainer.report({'x': 1}, observer)

    def test_report_scope(self):
        reporter = pytorch_trainer.Reporter()
        observation = {}

        with reporter:
            with pytorch_trainer.report_scope(observation):
                pytorch_trainer.report({'x': 1})

        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)
        self.assertNotIn('x', reporter.observation)


@backend.inject_backend_tests(
    ['test_basic', 'test_serialize_array_float', 'test_serialize_array_int'],
    [{}])
class TestSummary(unittest.TestCase):

    def setUp(self):
        self.summary = pytorch_trainer.reporter.Summary()

    def test_basic(self, backend_config):
        self.summary.add(backend_config.get_tensor(numpy.array(1, 'f')))
        self.summary.add(backend_config.get_tensor(numpy.array(-2, 'f')))

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, numpy.array(-0.5, 'f'))

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, numpy.array(-0.5, 'f'))
        testing.assert_allclose(std, numpy.array(1.5, 'f'))

    def test_int(self):
        self.summary.add(1)
        self.summary.add(2)
        self.summary.add(3)

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, 2)

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, 2)
        testing.assert_allclose(std, numpy.sqrt(2. / 3.))

    def test_float(self):
        self.summary.add(1.)
        self.summary.add(2.)
        self.summary.add(3.)

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, 2.)

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, 2.)
        testing.assert_allclose(std, numpy.sqrt(2. / 3.))

    def test_weight(self):
        self.summary.add(1., 0.5)
        self.summary.add(2., numpy.array(0.4))
        self.summary.add(3., torch.from_numpy(numpy.array(0.3)))

        mean = self.summary.compute_mean().array
        val = (1 * 0.5 + 2 * 0.4 + 3 * 0.3) / (0.5 + 0.4 + 0.3)
        testing.assert_allclose(mean, val)

    def check_serialize(self, value1, value2, value3):
        self.summary.add(value1)
        self.summary.add(value2)

        summary = pytorch_trainer.reporter.Summary()
        testing.save_and_load_pth(self.summary, summary)
        summary.add(value3)

        expected_mean = (value1 + value2 + value3).to(dtype=torch.float) / 3.
        expected_std = ((value1 ** 2 + value2 ** 2 + value3 ** 2)
            .to(dtype=torch.float) / 3. - expected_mean ** 2).sqrt()

        mean = summary.compute_mean()
        testing.assert_allclose(mean, expected_mean)

        mean, std = summary.make_statistics()
        testing.assert_allclose(mean, expected_mean)
        testing.assert_allclose(std, expected_std)

    def test_serialize_array_float(self, backend_config):
        self.check_serialize(
            backend_config.get_tensor(numpy.array(1.5, numpy.float32)),
            backend_config.get_tensor(numpy.array(2.0, numpy.float32)),
            # sum of the above two is non-integer
            backend_config.get_tensor(numpy.array(3.5, numpy.float32)))

    def test_serialize_array_int(self, backend_config):
        self.check_serialize(
            backend_config.get_tensor(numpy.array(1, numpy.int32)),
            backend_config.get_tensor(numpy.array(-2, numpy.int32)),
            backend_config.get_tensor(numpy.array(2, numpy.int32)))

    def test_serialize_scalar_float(self):
        self.check_serialize(
            1.5, 2.0,
            # sum of the above two is non-integer
            3.5)

    def test_serialize_scalar_int(self):
        self.check_serialize(1, -2, 2)

    def test_serialize_backward_compat(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # old version does not save anything
            torch.save(dict(dummy=0), f.name)
            with testing.assert_warns(UserWarning):
                self.summary.load_state_dict(torch.load(f.name))

        self.summary.add(2.)
        self.summary.add(3.)

        mean = self.summary.compute_mean()
        testing.assert_allclose(mean, 2.5)

        mean, std = self.summary.make_statistics()
        testing.assert_allclose(mean, 2.5)
        testing.assert_allclose(std, 0.5)


class TestDictSummary(unittest.TestCase):

    def setUp(self):
        self.summary = pytorch_trainer.reporter.DictSummary()

    def check(self, summary, data):
        mean = summary.compute_mean()
        self.assertEqual(set(mean.keys()), set(data.keys()))
        for name in data.keys():
            m = sum(data[name]) / float(len(data[name]))
            testing.assert_allclose(mean[name], m)

        stats = summary.make_statistics()
        self.assertEqual(
            set(stats.keys()),
            set(data.keys()).union(name + '.std' for name in data.keys()))
        for name in data.keys():
            m = sum(data[name]) / float(len(data[name]))
            s = numpy.sqrt(
                sum(x * x for x in data[name]) / float(len(data[name]))
                - m * m)
            testing.assert_allclose(stats[name], m)
            testing.assert_allclose(stats[name + '.std'], s)

    def test(self):
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
        self.summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

        self.check(self.summary, {
            'numpy': (3., 1., 2., 3.),
            'int': (1, 5, 6, 5),
            'float': (4., 9., 5., 8.),
        })

    def test_sparse(self):
        self.summary.add({'a': 3., 'b': 1.})
        self.summary.add({'a': 1., 'b': 5., 'c': 9.})
        self.summary.add({'b': 6.})
        self.summary.add({'a': 3., 'b': 5., 'c': 8.})

        self.check(self.summary, {
            'a': (3., 1., 3.),
            'b': (1., 5., 6., 5.),
            'c': (9., 8.),
        })

    def test_weight(self):
        self.summary.add({'a': (1., 0.5)})
        self.summary.add({'a': (2., numpy.array(0.4))})
        self.summary.add({'a': (3., torch.from_numpy(numpy.array(0.3)))})

        mean = self.summary.compute_mean()
        val = (1 * 0.5 + 2 * 0.4 + 3 * 0.3) / (0.5 + 0.4 + 0.3)
        testing.assert_allclose(mean['a'], val)

        with self.assertRaises(ValueError):
            self.summary.add({'a': (4., numpy.array([0.5]))})

        with self.assertRaises(ValueError):
            self.summary.add({'a': (4., torch.from_numpy(numpy.array([0.5])))})

    def test_serialize(self):
        self.summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
        self.summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
        self.summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})

        summary = pytorch_trainer.reporter.DictSummary()
        testing.save_and_load_pth(self.summary, summary)
        summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

        self.check(summary, {
            'numpy': (3., 1., 2., 3.),
            'int': (1, 5, 6, 5),
            'float': (4., 9., 5., 8.),
        })

    def test_serialize_names_with_slash(self):
        self.summary.add({'a/b': 3., '/a/b': 1., 'a/b/': 4.})
        self.summary.add({'a/b': 1., '/a/b': 5., 'a/b/': 9.})
        self.summary.add({'a/b': 2., '/a/b': 6., 'a/b/': 5.})

        summary = pytorch_trainer.reporter.DictSummary()
        testing.save_and_load_pth(self.summary, summary)
        summary.add({'a/b': 3., '/a/b': 5., 'a/b/': 8.})

        self.check(summary, {
            'a/b': (3., 1., 2., 3.),
            '/a/b': (1., 5., 6., 5.),
            'a/b/': (4., 9., 5., 8.),
        })

    def test_serialize_overwrite_different_names(self):
        self.summary.add({'a': 3., 'b': 1.})
        self.summary.add({'a': 1., 'b': 5.})

        summary = pytorch_trainer.reporter.DictSummary()
        summary.add({'c': 5.})
        testing.save_and_load_pth(self.summary, summary)

        self.check(summary, {
            'a': (3., 1.),
            'b': (1., 5.),
        })

    def test_serialize_overwrite_rollback(self):
        self.summary.add({'a': 3., 'b': 1.})
        self.summary.add({'a': 1., 'b': 5.})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            torch.save(self.summary.state_dict(), f.name)
            self.summary.add({'a': 2., 'b': 6., 'c': 5.})
            self.summary.add({'a': 3., 'b': 4., 'c': 6.})
            self.summary.load_state_dict(torch.load(f.name))

        self.summary.add({'a': 3., 'b': 5., 'c': 8.})

        self.check(self.summary, {
            'a': (3., 1., 3.),
            'b': (1., 5., 5.),
            'c': (8.,),
        })

    def test_serialize_backward_compat(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # old version does not save anything
            torch.save(dict(dummy=0), f.name)
            with testing.assert_warns(UserWarning):
                self.summary.load_state_dict(torch.load(f.name))

    def test_serialize_backward_compat_overwrite(self):
        self.summary.add({'a': 3., 'b': 1., 'c': 4.})
        self.summary.add({'a': 1., 'b': 5., 'c': 9.})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            # old version does not save anything
            torch.save(dict(dummy=0), f.name)
            with testing.assert_warns(UserWarning):
                self.summary.load_state_dict(torch.load(f.name))

        self.summary.add({'a': 9., 'b': 2.})
        self.summary.add({'a': 6., 'b': 5.})

        self.check(self.summary, {
            'a': (9., 6.),
            'b': (2., 5.),
        })


testing.run_module(__name__, __file__)
