"""Data loaders for EmoSurv and WESAD plus the unified dataset bundle.

See :mod:`empathic.data.emosurv` and :mod:`empathic.data.wesad` for the
dataset-specific logic, and :mod:`empathic.data.unified` for label harmonisation.
"""

from ..config import QUADRANTS  # noqa: F401
from .emosurv import load_emosurv  # noqa: F401
from .unified import (  # noqa: F401
    DatasetBundle,
    build_bundles,
    unify_quadrant_label,
)
from .wesad import load_wesad  # noqa: F401
