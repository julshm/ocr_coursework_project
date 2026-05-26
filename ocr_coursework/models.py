from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "text": self.text,
            "confidence": round(self.confidence, 2),
        }