from typing import List
import cv2
import numpy as np
from .models import TextRegion


class TextDetector:
    def detect_regions(self, detection_map: np.ndarray) -> List[TextRegion]:
        contours, _ = cv2.findContours(detection_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: List[TextRegion] = []
        img_h, img_w = detection_map.shape[:2]
        image_area = img_h * img_w

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 3000 or w < 120 or h < 30:
                continue
            if area > image_area * 0.08:
                continue
            if w > img_w * 0.90 and h < 120:
                continue
            if h > img_h * 0.08:
                continue
            if y < img_h * 0.02 and h < 80:
                continue
            if y + h > img_h * 0.98 and h < 80:
                continue
            regions.append(TextRegion(x=x, y=y, w=w, h=h))

        regions = sorted(regions, key=lambda r: (r.y, r.x))
        return self._merge_close_regions(regions)

    def _merge_close_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        if not regions:
            return []

        merged: List[TextRegion] = []
        current = regions[0]
        for nxt in regions[1:]:
            overlap_y = abs(nxt.y - current.y) < max(current.h, nxt.h) * 0.65
            close_x = nxt.x <= current.x + current.w + 60
            if overlap_y and close_x:
                x1 = min(current.x, nxt.x)
                y1 = min(current.y, nxt.y)
                x2 = max(current.x + current.w, nxt.x + nxt.w)
                y2 = max(current.y + current.h, nxt.y + nxt.h)
                current = TextRegion(x1, y1, x2 - x1, y2 - y1)
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
        return merged