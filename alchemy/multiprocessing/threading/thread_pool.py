# -*- coding: utf-8 -*-
from __future__ import absolute_import

import queue, threading

from .thread_worker import ThreadWorker


class ThreadPool(object):
  """Pool of threads consuming tasks from a queue

  Args:
    num_threads: int
      max number of threads to allow in the pool.
  """
  def __init__(self, num_threads):
    self.tasks = queue.Queue(num_threads)
    self.workers = []
    self.lock = threading.Lock()
    for _ in range(num_threads):
      self.workers.append(ThreadWorker(self.tasks))

  @property
  def done(self):
    return self.tasks.empty()

  def add_task(self, func, *args, **kargs):
    """
    Add a task to the queue

    Args:
      func: def|lambda
        task function to be called with `*args` and `**kargs`
    """
    self.tasks.put((func, args, kargs))

  def wait_completion(self):
    """Wait for completion of all the tasks in the queue.

    Returns:
      a list of lists containing returns from the task function.
    """
    self.tasks.join()
    results = []
    for worker in self.workers:
      results.append(worker.results)
      worker.results = []
    return results
