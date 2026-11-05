"""Socket-free, serial subset of Fiber used by the original POET driver.

Put this directory before the upstream checkout on ``sys.path``.  This module
does not implement multiprocessing: jobs execute eagerly in submission order.
Its scope is ``get_context().Manager().dict()`` and ``Pool.apply_async().get()``.

Each job temporarily installs a separate Python/NumPy random state, because
POET's simulator seeds both global generators inside a worker.  Consequently a
rollout cannot accidentally reset the master process's mutation/noise-seed RNG.
Manager values are pickled on write and unpickled on read, preserving the
copying boundary of the upstream manager rather than aliasing live environments.

No worker concurrency, process crashes, task cancellation, or timeout behavior
is emulated.  The upstream worker module has process-global state, so use one
active POET pool per interpreter.  Parallel execution can have a different
schedule; this adapter is not a claim of bit-for-bit parallel reproducibility.
"""

from collections.abc import MutableMapping
import pickle
import random

import numpy as np


class _LocalDict(MutableMapping):
    """Manager-like dictionary whose values cross a serialization boundary."""

    def __init__(self, *args, **kwargs):
        self._values = {}
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        return pickle.loads(self._values[key])

    def __setitem__(self, key, value):
        self._values[key] = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def __delitem__(self, key):
        del self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class Manager:
    """Construct local serialized dictionaries without starting a server."""

    def dict(self, *args, **kwargs):
        return _LocalDict(*args, **kwargs)

    def shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.shutdown()


class _Result:
    """Already-completed result with the get/ready/successful pool interface."""

    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def get(self, timeout=None):
        if self._error is not None:
            raise self._error
        return self._value

    def ready(self):
        return True

    def successful(self):
        return self._error is None

    def wait(self, timeout=None):
        pass


class Pool:
    """Execute the original worker callbacks serially with isolated RNG state.

    ``processes`` is accepted for upstream API compatibility but does not create
    workers.  ``apply_async`` completes before returning its result object;
    eager execution also ensures that each task observes the theta published
    when the job was submitted, before a later optimizer broadcast replaces it.
    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        if processes is not None and processes < 1:
            raise ValueError("Number of processes must be at least 1")
        if maxtasksperchild is not None:
            raise NotImplementedError("Serial POET adapter does not recycle workers")
        self.processes = processes
        self._closed = False
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        if initializer is not None:
            self._run(initializer, initargs, {})

    def _run(self, function, args, kwargs):
        master_python = random.getstate()
        master_numpy = np.random.get_state()
        random.setstate(self._python_state)
        np.random.set_state(self._numpy_state)
        try:
            return function(*args, **kwargs)
        finally:
            self._python_state = random.getstate()
            self._numpy_state = np.random.get_state()
            random.setstate(master_python)
            np.random.set_state(master_numpy)

    def apply_async(self, func, args=(), kwds=None, callback=None,
                    error_callback=None):
        if self._closed:
            raise ValueError("Pool not running")
        try:
            value = self._run(func, args, {} if kwds is None else kwds)
        except Exception as error:
            result = _Result(error=error)
            if error_callback is not None:
                error_callback(error)
        else:
            result = _Result(value=value)
            if callback is not None:
                callback(value)
        return result

    def close(self):
        self._closed = True

    def terminate(self):
        self.close()

    def join(self):
        if not self._closed:
            raise ValueError("Pool is still running")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.terminate()


class _Context:
    Manager = Manager
    Pool = Pool


def get_context(method=None):
    if method not in (None, "spawn"):
        raise ValueError("Serial POET adapter only accepts the spawn context")
    return _Context()
