"""Root conftest: patches missing dependencies for the test environment.

The system Python may lack optional/heavy packages (lpips, torchvision).
This conftest stubs them so the test suite can import src.* modules
without error.
"""

import builtins
import sys
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub missing optional packages before any src imports
# ---------------------------------------------------------------------------

def _stub_missing_package(*names: str) -> None:
    """Insert MagicMocks into sys.modules for packages that aren't installed."""
    for name in names:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = MagicMock()


def _install_import_hook() -> None:
    """Auto-stub sub-module imports of already-mocked root packages.

    When a root package (e.g. ``torchvision``) is mocked, imports like
    ``from torchvision.models import X`` would fail because MagicMock
    isn't a real package. This hook intercepts such imports and returns
    MagicMock sub-modules automatically.
    """
    _original_import = builtins.__import__
    _stubbed_roots: set[str] = set()

    def _hook_import(name, *args, **kwargs):
        try:
            return _original_import(name, *args, **kwargs)
        except (ImportError, ModuleNotFoundError):
            root = name.split(".")[0]
            if root in _stubbed_roots or isinstance(sys.modules.get(root), MagicMock):
                _stubbed_roots.add(root)
                if name not in sys.modules:
                    sys.modules[name] = MagicMock()
                return sys.modules[name]
            raise

    builtins.__import__ = _hook_import


# Heavy optional dependencies that may not be installed in the test env
_stub_missing_package("lpips", "torchvision")
_install_import_hook()


# ---------------------------------------------------------------------------
# Ensure MultiModalDataset.get_modality_metadata exists (added in 0.4.1)
# ---------------------------------------------------------------------------

def _ensure_get_modality_metadata() -> None:
    """Patch get_modality_metadata onto MultiModalDataset if missing."""
    from euler_loading import MultiModalDataset

    if hasattr(MultiModalDataset, "get_modality_metadata"):
        return

    def get_modality_metadata(self, modality_name: str) -> dict[str, Any]:
        return self._index_outputs.get(modality_name, {}).get("meta", {})

    MultiModalDataset.get_modality_metadata = get_modality_metadata


_ensure_get_modality_metadata()
