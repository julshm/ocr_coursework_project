from typing import Dict
import cv2
import numpy as np


class ImagePreprocessor:
    def crop_document_margins(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        top = int(h * 0.015)
        bottom = int(h * 0.03)
        left = int(w * 0.02)
        right = int(w * 0.02)
        return image[top:h - bottom, left:w - right]

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Зображення не завантажено")
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def upscale(self, gray: np.ndarray, scale: float = 2.0) -> np.ndarray:
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    def binarize_otsu(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def binarize_adaptive(self, gray: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)

    def binarize_document(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(inv > 0))
        if len(coords) < 100:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if abs(angle) < 0.3:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def morphology_for_detection(self, binary: np.ndarray) -> np.ndarray:
        inv = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 9))
        return cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)

    def preprocess(self, image: np.ndarray, is_document: bool) -> Dict[str, np.ndarray]:
        base = self.crop_document_margins(image) if is_document else image
        gray = self.to_grayscale(base)
        contrast = self.enhance_contrast(gray)
        upscaled = self.upscale(contrast, scale=2.0)
        denoised = self.denoise(upscaled)
        binary_otsu = self.binarize_otsu(denoised)
        binary_adaptive = self.binarize_adaptive(denoised)
        binary_document = self.binarize_document(denoised)
        deskewed = self.deskew(binary_document)
        detection_map = self.morphology_for_detection(deskewed)
        return {
            "base": base,
            "gray": gray,
            "contrast": contrast,
            "upscaled": upscaled,
            "denoised": denoised,
            "binary_otsu": binary_otsu,
            "binary_adaptive": binary_adaptive,
            "binary_document": binary_document,
            "deskewed": deskewed,
            "detection_map": detection_map,
        }