"""Progress logging and the run-wide warning list.

warn() both echoes and accumulates, so write_config can replay every warning a
run produced. The elapsed clock starts when this module is first imported.
"""

import time


WARNINGS = []
_T0 = time.time()


def log(msg):
    """Print a timestamped progress line."""
    print(f'[{time.time() - _T0:7.1f}s] {msg}', flush=True)


def warn(msg):
    """Record a warning for the config file and echo it."""
    WARNINGS.append(msg)
    print(f'  ! {msg}', flush=True)
