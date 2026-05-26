import csv
import os
from typing import Dict, List


def save_batch_reports(output_dir: str, reports: List[Dict], filename_prefix: str = "batch_report") -> Dict[str, str]:
    csv_path = os.path.join(output_dir, f"{filename_prefix}.csv")
    txt_path = os.path.join(output_dir, f"{filename_prefix}.txt")

    fieldnames = [
        "image_path", "mode_used", "regions_count", "avg_region_confidence", "total_ms",
        "char_accuracy", "cer", "word_accuracy", "wer", "recognized_text_file",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            metrics = report.get("metrics") or {}
            writer.writerow({
                "image_path": report.get("image_path", ""),
                "mode_used": report.get("mode_used", ""),
                "regions_count": report.get("regions_count", 0),
                "avg_region_confidence": report.get("average_region_confidence", 0.0),
                "total_ms": report.get("timings_ms", {}).get("total", 0.0),
                "char_accuracy": metrics.get("char_accuracy", ""),
                "cer": metrics.get("cer", ""),
                "word_accuracy": metrics.get("word_accuracy", ""),
                "wer": metrics.get("wer", ""),
                "recognized_text_file": report.get("output_files", {}).get("recognized_text", ""),
            })

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("ПІДСУМКОВИЙ ЗВІТ ПО СЕРІЇ ЕКСПЕРИМЕНТІВ OCR\n")
        f.write("=" * 70 + "\n\n")
        for idx, report in enumerate(reports, start=1):
            metrics = report.get("metrics") or {}
            f.write(f"{idx}. Файл: {report.get('image_path', '')}\n")
            f.write(f"   Режим: {report.get('mode_used', '')}\n")
            f.write(f"   Кількість областей: {report.get('regions_count', 0)}\n")
            f.write(f"   Середня confidence: {report.get('average_region_confidence', 0.0)}\n")
            f.write(f"   Час виконання, мс: {report.get('timings_ms', {}).get('total', 0.0)}\n")
            if metrics:
                f.write(f"   Char accuracy: {metrics.get('char_accuracy', '')}\n")
                f.write(f"   CER: {metrics.get('cer', '')}\n")
                f.write(f"   Word accuracy: {metrics.get('word_accuracy', '')}\n")
                f.write(f"   WER: {metrics.get('wer', '')}\n")
            else:
                f.write("   Метрики: ground truth не задано\n")
            f.write(f"   TXT : {report.get('output_files', {}).get('recognized_text', '')}\n")
            f.write("-" * 70 + "\n")

    return {"csv": csv_path, "txt": txt_path}