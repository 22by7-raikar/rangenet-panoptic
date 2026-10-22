# Replacement for imp.load_source, which was removed in Python 3.12.

import importlib.util


def load_source(name, path):
  """Import the Python file at ``path`` as a module called ``name``."""
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module
