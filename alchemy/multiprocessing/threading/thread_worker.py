# -*- coding: utf-8 -*-
import threading


class ThreadWorker(threading.Thread):
  """Thread executing tasks from a given task `Queue`.

  Args:
    tasks: `queue.Queue`
      Queue of tasks to consume be consumed.
  """
  def __init__(self, tasks):
    threading.Thread.__init__(self)
    self.tasks = tasks
    self.daemon = True
    self.results = []
    self.start()

  def run(self):
    """Run a task on the queue and append what is returned from the task to the `self.results`."""
    while True:
      func, args, kargs = self.tasks.get()
      try:
        self.results.append(func(*args, **kargs))
      except Exception as e:
        raise ValueError(str(e))
      finally:
        self.tasks.task_done()
