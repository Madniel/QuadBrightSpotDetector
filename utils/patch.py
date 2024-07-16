from dataclasses import dataclass
from typing import Tuple


@dataclass
class Patch:
    center: Tuple[int, int]
    brightness: float
