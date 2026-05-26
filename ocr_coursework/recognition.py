from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pytesseract
from .models import TextRegion
from .text_utils import (
    clean_text,
    looks_like_author_line,
    normalize_author_line,
    normalize_line_case,
    normalize_sentence_like_line,
)


class OCRRecognizer:
    def __init__(self, tesseract_cmd: str = None, lang: str = "ukr"):
        self.lang = lang
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @staticmethod
    def _safe_conf(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1.0

    def _extract_lines(self, image: np.ndarray, config: str) -> List[Tuple[str, float]]:
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        line_map: Dict[Tuple[int, int, int], List[str]] = {}
        conf_map: Dict[Tuple[int, int, int], List[float]] = {}
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            conf = self._safe_conf(data["conf"][i])
            if not text or conf < 0:
                continue

            key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
            line_map.setdefault(key, []).append(text)
            conf_map.setdefault(key, []).append(conf)

        lines: List[Tuple[str, float]] = []
        for key in sorted(line_map.keys()):
            line = " ".join(line_map[key]).strip()
            if line:
                avg_conf = float(np.mean(conf_map[key])) if conf_map[key] else 0.0
                lines.append((line, avg_conf))
        return lines

    def _ocr_data_to_text_and_conf(self, image: np.ndarray, config: str) -> Tuple[str, float]:
        lines = self._extract_lines(image, config)
        if not lines:
            return "", 0.0
        text = "\n".join(line for line, _ in lines).strip()
        conf = float(np.mean([c for _, c in lines])) if lines else 0.0
        return text, conf

    def recognize_full_image(self, images: List[np.ndarray]) -> str:
        configs = [
            "--oem 3 --psm 6 -c preserve_interword_spaces=1",
            "--oem 3 --psm 4 -c preserve_interword_spaces=1",
            "--oem 3 --psm 3 -c preserve_interword_spaces=1",
        ]

        best_text = ""
        best_score = -1.0
        for image in images:
            for config in configs:
                text, avg_conf = self._ocr_data_to_text_and_conf(image, config)
                if not text:
                    continue
                non_space = sum(1 for ch in text if not ch.isspace())
                letters = sum(1 for ch in text if ch.isalpha())
                digits = sum(1 for ch in text if ch.isdigit())
                info_ratio = (letters + digits) / max(non_space, 1)
                line_count = len([line for line in text.splitlines() if line.strip()])
                score = avg_conf + info_ratio * 35.0 + min(line_count, 80) * 0.15
                if score > best_score:
                    best_score = score
                    best_text = text
        return best_text.strip()

    def recognize_document_header(self, images: List[np.ndarray]) -> str:
        if not images:
            return ""

        title_crops: List[np.ndarray] = []
        subtitle_crops: List[np.ndarray] = []
        for image in images:
            h, w = image.shape[:2]
            title_crops.extend([
                image[int(h * 0.11):int(h * 0.17), int(w * 0.22):int(w * 0.78)],
                image[int(h * 0.10):int(h * 0.18), int(w * 0.18):int(w * 0.82)],
            ])
            subtitle_crops.append(image[int(h * 0.16):int(h * 0.23), int(w * 0.12):int(w * 0.88)])

        title_line = self._best_single_line(
            title_crops,
            ["--oem 3 --psm 7 -c preserve_interword_spaces=1", "--oem 3 --psm 6 -c preserve_interword_spaces=1"],
            keyword_bonus=("догов", "ухвал", "наказ", "акт"),
        )
        subtitle_line = self._best_single_line(
            subtitle_crops,
            ["--oem 3 --psm 6 -c preserve_interword_spaces=1", "--oem 3 --psm 7 -c preserve_interword_spaces=1"],
            keyword_bonus=("про надання", "послуг", "вивезення", "обшуку"),
        )

        parts = [line for line in [self._postprocess_header_title(title_line), subtitle_line] if line]
        return "\n".join(parts).strip()

    def _best_single_line(self, crops: List[np.ndarray], configs: List[str], keyword_bonus: Tuple[str, ...]) -> str:
        best_line = ""
        best_score = -1.0
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            for config in configs:
                text, avg_conf = self._ocr_data_to_text_and_conf(crop, config)
                if not text:
                    continue
                first_line = text.splitlines()[0].strip()
                if not first_line:
                    continue
                low = first_line.lower()
                non_space = sum(1 for ch in first_line if not ch.isspace())
                letters = sum(1 for ch in first_line if ch.isalpha())
                digits = sum(1 for ch in first_line if ch.isdigit())
                info_ratio = (letters + digits) / max(non_space, 1)
                score = avg_conf + info_ratio * 30.0
                if any(key in low for key in keyword_bonus):
                    score += 35.0
                if "№" in first_line or any(ch.isdigit() for ch in first_line):
                    score += 15.0
                if score > best_score:
                    best_score = score
                    best_line = first_line
        return best_line.strip()

    @staticmethod
    def _postprocess_header_title(line: str) -> str:
        if not line:
            return line
        import re

        line = line.strip()
        line = re.sub(r'\b[Лл][Ее]\b', '№', line)
        line = re.sub(r'\b[Лл][оО]\b', '№', line)
        line = re.sub(r'\bN[o0О°]\b', '№', line, flags=re.IGNORECASE)
        if "догов" in line.lower():
            chars = [ch.upper() if ch.isalpha() else ch for ch in line]
            line = "".join(chars)
            line = re.sub(r"ДОГОВ[ІIЇЇРP]+", "ДОГОВІР", line)
        line = re.sub(r'№\s*', '№ ', line)
        line = re.sub(r'[ ]{2,}', ' ', line)
        return line.strip()

    def recognize_receipt_text(self, images: List[np.ndarray]) -> str:
        configs = [
            "--oem 3 --psm 6 -c preserve_interword_spaces=1",
            "--oem 3 --psm 4 -c preserve_interword_spaces=1",
        ]
        best_text = ""
        best_score = -1.0
        expanded_images: List[np.ndarray] = []

        for image in images:
            expanded_images.append(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
            expanded_images.append(adaptive)

        for image in expanded_images:
            for config in configs:
                text, avg_conf = self._ocr_data_to_text_and_conf(image, config)
                if not text:
                    continue
                text = clean_text(text)
                low = text.lower()
                bonus = sum(2.0 for key in [
                    "квитанц", "термінал", "дата", "сума", "отримувач", "платник", "грн",
                    "операції", "телефон", "єдрпоу", "контрактовий", "ліцензії", "банк", "р/р",
                ] if key in low)
                non_space = sum(1 for ch in text if not ch.isspace())
                letters = sum(1 for ch in text if ch.isalpha())
                digits = sum(1 for ch in text if ch.isdigit())
                info_ratio = (letters + digits) / max(non_space, 1)
                score = avg_conf + info_ratio * 30.0 + bonus
                if score > best_score:
                    best_score = score
                    best_text = text
        return best_text.strip()

    def recognize_region(self, image: np.ndarray, region: TextRegion, is_document: bool = False) -> TextRegion:
        pad = 12
        h_img, w_img = image.shape[:2]
        x1 = max(0, region.x - pad)
        y1 = max(0, region.y - pad)
        x2 = min(w_img, region.x + region.w + pad)
        y2 = min(h_img, region.y + region.h + pad)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return region

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()
        roi_up = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, roi_bin = cv2.threshold(roi_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        raw_text, confidence = self._ocr_data_to_text_and_conf(roi_bin, "--oem 3 --psm 7 -c preserve_interword_spaces=1")
        raw_text = normalize_line_case(raw_text)

        if not is_document:
            candidate = normalize_sentence_like_line(raw_text)
            raw_text = normalize_author_line(raw_text) if looks_like_author_line(raw_text) else candidate
        else:
            if looks_like_author_line(raw_text):
                raw_text = normalize_author_line(raw_text)
            elif confidence < 85:
                raw_text = normalize_sentence_like_line(raw_text)

        region.text = raw_text.strip()
        region.confidence = confidence
        return region