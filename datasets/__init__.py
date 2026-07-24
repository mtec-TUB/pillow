"""
Discover Dataset modules without importing them.

Each dataset script is only imported (and its own imports checked) once it is
actually requested via `datasets.registry.get_dataset()`, e.g. because it is
the dataset chosen in config.yaml. This avoids requiring every dataset's
dependencies to be installed just to run one of them.
"""

import re
from pathlib import Path

from datasets.registry import DatasetRegistry

_REGISTER_RE = re.compile(r'@register_dataset\(\s*["\']([^"\']+)["\']\s*\)')
_SKIP_MODULES = {"__init__", "base", "registry"}

for _path in sorted(Path(__path__[0]).glob("*.py")):
    if _path.stem in _SKIP_MODULES:
        continue
    _match = _REGISTER_RE.search(_path.read_text())
    if _match:
        DatasetRegistry.register_module(_match.group(1), _path.stem)
