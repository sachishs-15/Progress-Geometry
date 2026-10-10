"""Importing this package registers every model plugin found in models/*/model.py.

To add a new encoder: create `models/<name>/model.py` implementing `FrozenEncoder`
(common/base.py) and decorate the class with `@register`, then add the import below.
"""
from models.clip import model as _clip  # noqa: F401
from models.dinov2 import model as _dinov2  # noqa: F401
from models.dinov3 import model as _dinov3  # noqa: F401
from models.levljepa import model as _levljepa  # noqa: F401
from models.lift import model as _lift  # noqa: F401
from models.siglip import model as _siglip  # noqa: F401
from models.vjepa2 import model as _vjepa2  # noqa: F401
from models.vjepa2_1 import model as _vjepa2_1  # noqa: F401
