import argparse
import os
from ocr_coursework.io_utils import load_text_file, resolve_ground_truth_path
from ocr_coursework.pipeline import OCRPipeline, process_folder
from ocr_coursework.tesseract_utils import ensure_tesseract_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Виявлення та розпізнавання тексту на зображеннях за допомогою OpenCV та pytesseract"
    )
    parser.add_argument("--input", required=True, help="Шлях до зображення або каталогу із зображеннями")
    parser.add_argument("--output", default="output", help="Каталог для збереження результатів")
    parser.add_argument("--tesseract-cmd", default=None, help="Повний шлях до tesseract.exe")
    parser.add_argument("--lang", default="ukr", help="Мова OCR: ukr, eng, ukr+eng")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "document", "receipt", "simple"],
        help="Режим обробки: auto/document/receipt/simple",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Шлях до еталонного txt-файлу або до папки з еталонними txt-файлами",
    )
    parser.add_argument(
        "--batch-report-name",
        default="batch_report",
        help="Базова назва CSV/TXT-звіту при обробці папки",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available_langs = ensure_tesseract_ready(args.tesseract_cmd, args.lang)
    print("Tesseract OCR доступний.")
    print("Доступні мови:", ", ".join(available_langs))

    if os.path.isdir(args.input):
        process_folder(
            input_dir=args.input,
            output_dir=args.output,
            tesseract_cmd=args.tesseract_cmd,
            lang=args.lang,
            mode=args.mode,
            ground_truth=args.ground_truth,
            batch_report_name=args.batch_report_name,
        )
        return

    os.makedirs(args.output, exist_ok=True)

    gt_path = resolve_ground_truth_path(args.input, args.ground_truth)
    if args.ground_truth and gt_path is None:
        raise FileNotFoundError(
            f"Не знайдено ground truth: {args.ground_truth}\n"
            f"Перевір, чи існує файл або папка, і чи ім'я txt збігається з ім'ям зображення."
        )

    gt_text = load_text_file(gt_path) if gt_path else None
    if gt_path:
        print("Ground truth:", gt_path)

    pipeline = OCRPipeline(tesseract_cmd=args.tesseract_cmd, lang=args.lang)
    report = pipeline.process(
        image_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        ground_truth_text=gt_text,
        ground_truth_path=gt_path,
    )

    print("Файл:", report["image_path"])
    print("Режим:", report["mode_used"])
    print("Кількість знайдених текстових областей:", report["regions_count"])
    print("Час виконання, мс:", report["timings_ms"]["total"])
    print("Результат OCR:")
    print(report["recognized_text"])

    if report["regions"]:
        print("\nДеталі по областях:")
        for idx, region in enumerate(report["regions"], start=1):
            print(
                f"{idx}. x={region['x']}, y={region['y']}, w={region['w']}, h={region['h']}, "
                f"conf={region['confidence']}%, text={region['text']}"
            )

    if report.get("metrics"):
        print("\nОцінка точності OCR:")
        print("ground_truth_path:", report["ground_truth_path"])
        print("char_accuracy:", report["metrics"]["char_accuracy"])
        print("CER:", report["metrics"]["cer"])
        print("word_accuracy:", report["metrics"]["word_accuracy"])
        print("WER:", report["metrics"]["wer"])
    else:
        print("\nОцінка точності OCR не виконувалась: ground truth не задано.")

    print("TXT-результат:", report["output_files"]["recognized_text"])


if __name__ == "__main__":
    main()