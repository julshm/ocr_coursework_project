from typing import List, Optional
import pytesseract


def ensure_tesseract_ready(tesseract_cmd: Optional[str], requested_lang: str) -> List[str]:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Не вдалося запустити Tesseract OCR. Перевір, чи встановлено Tesseract і чи правильно передано --tesseract-cmd."
        ) from e

    try:
        available_langs = pytesseract.get_languages(config="")
    except Exception as e:
        raise RuntimeError("Tesseract встановлено, але не вдалося отримати список мов.") from e

    requested = [lang.strip() for lang in requested_lang.split("+") if lang.strip()]
    missing = [lang for lang in requested if lang not in available_langs]
    if missing:
        raise RuntimeError(
            "Відсутні мовні пакети Tesseract: "
            + ", ".join(missing)
            + ". Доступні: "
            + ", ".join(sorted(available_langs))
        )

    return sorted(available_langs)