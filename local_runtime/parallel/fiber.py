"""Multi-process stand-in for Fiber, which mirrors the multiprocessing API.

Put this directory before the upstream checkout on ``sys.path``. Fork lets every
worker share the master's noise table copy-on-write instead of regenerating it.
Worker scheduling differs from a Fiber cluster, so runs make no claim of being
bit-for-bit identical to upstream.
"""
import multiprocessing


def get_context(method=None):
    return multiprocessing.get_context("fork")
