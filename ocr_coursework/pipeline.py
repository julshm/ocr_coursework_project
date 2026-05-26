import os
import time
from typing import Dict, List, Optional
import cv2
from .detection import TextDetector
from .io_utils import load_text_file, resolve_ground_truth_path
from .metrics import evaluate_ocr_against_ground_truth
from .models import TextRegion
from .preprocessing import ImagePreprocessor
from .recognition import OCRRecognizer
from .reporting import save_batch_reports
from .text_utils import clean_text, normalize_document_symbols, normalize_receipt_text


class OCRPipeline:
    def __init__(self, tesseract_cmd: str = None, lang: str = "ukr"):
        self.preprocessor = ImagePreprocessor()
        self.detector = TextDetector()
        self.recognizer = OCRRecognizer(tesseract_cmd=tesseract_cmd, lang=lang)
        self.lang = lang

    @staticmethod
    def is_document_like(image) -> bool:
        h, w = image.shape[:2]
        return h / max(w, 1) > 1.15

    @staticmethod
    def is_receipt_like(image) -> bool:
        h, w = image.shape[:2]
        aspect = h / max(w, 1)
        return 1.2 <= aspect <= 2.3 and w <= 900

    def resolve_mode(self, image, mode: str) -> str:
        if mode != "auto":
            return mode
        if self.is_receipt_like(image):
            return "receipt"
        if self.is_document_like(image):
            return "document"
        return "simple"

    def filter_recognized_regions(self, regions: List[TextRegion], is_document: bool) -> List[TextRegion]:
        filtered: List[TextRegion] = []
        for region in regions:
            text = region.text.strip()
            if not text:
                continue
            if len(text) <= 3 and region.confidence < 65:
                continue
            if region.h < 40 and len(text) <= 5:
                continue
            if is_document and region.confidence < 55:
                continue
            filtered.append(region)
        return filtered

    @staticmethod
    def merge_header_with_body(header_text: str, body_text: str) -> str:
        if not body_text:
            return header_text.strip()

        body_lines = [line.rstrip() for line in body_text.splitlines()]
        while body_lines:
            line = body_lines[0].strip()
            pure = "".join(ch for ch in line if ch.isalnum())
            if len(pure) < 6:
                body_lines.pop(0)
            else:
                break

        body_text = "\n".join(body_lines).strip()
        if not header_text:
            return body_text

        body_start = "\n".join(body_text.splitlines()[:8]).lower()
        header_lines = [line.strip() for line in header_text.splitlines() if line.strip()]
        lines_to_add = [line for line in header_lines if line.lower().strip() not in body_start]
        return ("\n".join(lines_to_add) + "\n" + body_text).strip() if lines_to_add else body_text


    def save_intermediate_results(self, output_dir: str, base_name: str, preprocessed: Dict, annotated, full_text: str) -> Dict[str, str]:
        paths = {
            "gray": os.path.join(output_dir, f"{base_name}_01_gray.png"),
            "denoised": os.path.join(output_dir, f"{base_name}_04_denoised.png"),
            "binary_document": os.path.join(output_dir, f"{base_name}_07_binary_document.png"),
            "deskewed": os.path.join(output_dir, f"{base_name}_08_deskewed.png"),
            "detection_map": os.path.join(output_dir, f"{base_name}_09_detection_map.png"),
            "annotated": os.path.join(output_dir, f"{base_name}_10_detected_regions.png"),
            "recognized_text": os.path.join(output_dir, f"{base_name}_recognized_text.txt"),
        }

        cv2.imwrite(paths["gray"], preprocessed["gray"])
        cv2.imwrite(paths["denoised"], preprocessed["denoised"])
        cv2.imwrite(paths["binary_document"], preprocessed["binary_document"])
        cv2.imwrite(paths["deskewed"], preprocessed["deskewed"])
        cv2.imwrite(paths["detection_map"], preprocessed["detection_map"])
        cv2.imwrite(paths["annotated"], annotated)
        with open(paths["recognized_text"], "w", encoding="utf-8") as f:
            f.write(full_text)
        return paths

    def draw_regions(self, image, regions: List[TextRegion]):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for region in regions:
            cv2.rectangle(image, (region.x, region.y), (region.x + region.w, region.y + region.h), (0, 255, 0), 2)
            label = f"{region.confidence:.1f}%"
            cv2.putText(image, label, (region.x, max(20, region.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def process(self, image_path: str, output_dir: str, mode: str = "auto", ground_truth_text: Optional[str] = None,
                ground_truth_path: Optional[str] = None) -> Dict:
        total_start = time.perf_counter()
        image = cv2.imread(image_path)
        if image is None: raise FileNotFoundError(f"Не вдалося відкрити файл: {image_path}")

        mode_used = self.resolve_mode(image, mode)
        is_receipt, is_document = mode_used == "receipt", mode_used == "document"

        preprocess_start = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(image, is_document=is_document)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        ocr_start = time.perf_counter()
        if is_receipt:
            receipt_gray = self.preprocessor.to_grayscale(image)
            receipt_contrast = self.preprocessor.enhance_contrast(receipt_gray)
            receipt_upscaled = self.preprocessor.upscale(receipt_contrast, scale=3.0)
            _, receipt_otsu = cv2.threshold(receipt_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            full_text = self.recognizer.recognize_receipt_text(
                [receipt_gray, receipt_contrast, receipt_upscaled, receipt_otsu])
        else:
            full_text = self.recognizer.recognize_full_image(
                [preprocessed["binary_document"], preprocessed["deskewed"], preprocessed["contrast"]] if is_document
                else [preprocessed["contrast"], preprocessed["denoised"], preprocessed["binary_otsu"],
                      preprocessed["binary_document"], preprocessed["deskewed"]]
            )
            if is_document:
                header_text = self.recognizer.recognize_document_header([preprocessed["binary_document"], preprocessed["deskewed"]])
                if header_text and ("догов" in header_text.lower() or "ухвал" in header_text.lower() or "акт" in header_text.lower() or "наказ" in header_text.lower()):
                    full_text = self.merge_header_with_body(header_text, full_text)
        ocr_ms = (time.perf_counter() - ocr_start) * 1000.0

        regions_start = time.perf_counter()
        recognized_regions: List[TextRegion] = []
        if not is_receipt:
            regions = self.detector.detect_regions(preprocessed["detection_map"])
            if is_document: regions = sorted(sorted(regions, key=lambda r: r.w * r.h, reverse=True)[:8],
                                             key=lambda r: (r.y, r.x))
            for region in regions:
                recognized = self.recognizer.recognize_region(preprocessed["deskewed"], region, is_document=is_document)
                if recognized.text: recognized_regions.append(recognized)
            recognized_regions = self.filter_recognized_regions(recognized_regions, is_document=is_document)
            if is_document: recognized_regions = sorted(recognized_regions, key=lambda r: (r.y, r.x))
        regions_ms = (time.perf_counter() - regions_start) * 1000.0

        region_text = "\n".join(r.text for r in recognized_regions).strip()
        avg_conf = sum(r.confidence for r in recognized_regions) / len(
            recognized_regions) if recognized_regions else 0.0
        if mode_used == "simple" and 3 <= len(recognized_regions) <= 15 and len(
            region_text) > 30 and avg_conf >= 70: full_text = region_text

        full_text = normalize_document_symbols(clean_text(full_text))
        if is_receipt: full_text = normalize_receipt_text(full_text)

        eval_start = time.perf_counter()
        metrics = evaluate_ocr_against_ground_truth(full_text,
                                                    ground_truth_text) if ground_truth_text is not None else None
        eval_ms = (time.perf_counter() - eval_start) * 1000.0

        annotated = self.draw_regions(preprocessed["upscaled"].copy(), recognized_regions)
        output_files = self.save_intermediate_results(output_dir, base_name, preprocessed, annotated, full_text)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        report = {
            "image_path": image_path,
            "mode_requested": mode,
            "mode_used": mode_used,
            "language": self.lang,
            "ground_truth_path": ground_truth_path,
            "regions_count": len(recognized_regions),
            "average_region_confidence": round(avg_conf, 4),
            "timings_ms": {
                "preprocessing": round(preprocess_ms, 3),
                "ocr": round(ocr_ms, 3),
                "regions": round(regions_ms, 3),
                "evaluation": round(eval_ms, 3),
                "total": round(total_ms, 3),
            },
            "regions": [r.to_dict() for r in recognized_regions],
            "recognized_text": full_text,
            "metrics": metrics,
            "output_files": output_files,
        }
        return report


def process_folder(
    input_dir: str,
    output_dir: str,
    tesseract_cmd: str = None,
    lang: str = "ukr",
    mode: str = "auto",
    ground_truth: Optional[str] = None,
    batch_report_name: str = "batch_report",
) -> None:
    pipeline = OCRPipeline(tesseract_cmd=tesseract_cmd, lang=lang)
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted([
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.splitext(name.lower())[1] in supported_ext
    ])

    if not files:
        print(f"У каталозі {input_dir} не знайдено підтримуваних зображень")
        return

    os.makedirs(output_dir, exist_ok=True)
    total_start = time.perf_counter()
    reports: List[Dict] = []

    for file_path in files:
        print(f"Обробка: {file_path}")
        try:
            gt_path = resolve_ground_truth_path(file_path, ground_truth)
            gt_text = load_text_file(gt_path) if gt_path else None
            report = pipeline.process(file_path, output_dir, mode, gt_text, gt_path)
            reports.append(report)
            print("Кількість текстових областей:", report["regions_count"])
            print("Режим:", report["mode_used"])
            print("Час виконання, мс:", report["timings_ms"]["total"])
            print("Розпізнаний текст:")
            print(report["recognized_text"])
            if report.get("metrics"):
                print("Метрики:")
                print("  char_accuracy =", report["metrics"]["char_accuracy"])
                print("  CER           =", report["metrics"]["cer"])
                print("  word_accuracy =", report["metrics"]["word_accuracy"])
                print("  WER           =", report["metrics"]["wer"])
            print("-" * 80)
        except Exception as e:
            print(f"Помилка при обробці {file_path}: {e}")

    batch_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
    batch_report_paths = save_batch_reports(output_dir, reports, filename_prefix=batch_report_name)
    print("\nСЕРІЙНА ОБРОБКА ЗАВЕРШЕНА")
    print(f"Усього файлів успішно оброблено: {len(reports)}")
    print(f"Загальний час, мс: {batch_elapsed_ms:.2f}")
    print("CSV-звіт:", batch_report_paths["csv"])
    print("TXT-звіт:", batch_report_paths["txt"])