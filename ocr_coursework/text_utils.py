import re
from typing import List


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned: List[str] = []

    for line in lines:
        if not line:
            cleaned.append("")
            continue

        non_space = sum(1 for ch in line if not ch.isspace())
        letters = sum(1 for ch in line if ch.isalpha())
        digits = sum(1 for ch in line if ch.isdigit())
        informative_ratio = (letters + digits) / max(non_space, 1)

        if len(line) > 45 and informative_ratio < 0.30:
            continue

        unique_chars = set(ch.lower() for ch in line if not ch.isspace())
        if len(line) > 50 and len(unique_chars) < 6:
            continue

        cleaned.append(" ".join(line.split()))

    result: List[str] = []
    prev_empty = False
    for line in cleaned:
        if not line:
            if not prev_empty:
                result.append("")
            prev_empty = True
        else:
            result.append(line)
            prev_empty = False

    return "\n".join(result).strip()


def normalize_document_symbols(text: str) -> str:
    if not text:
        return text

    text = re.sub(
        r'(?<!\w)([Лл][Ее])(?=\s+[A-ZА-ЯІЇЄҐ0-9][A-ZА-ЯІЇЄҐ0-9\-]*)',
        '№',
        text,
    )
    text = re.sub(
        r'(?<!\w)(N[o0О°])(?=\s*[A-ZА-ЯІЇЄҐ0-9][A-ZА-ЯІЇЄҐ0-9\-]*)',
        '№',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'№\s*', '№ ', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()


def normalize_receipt_text(text: str) -> str:
    if not text:
        return text

    replacements = [
        (r"\bГОВ\b", "ТОВ"),
        (r"\bГелефон\b", "Телефон"),
        (r"\b[оО]ДРПОУ\b", "ЄДРПОУ"),
        (r"\bД операції\b", "ID операції"),
        (r"\b[МмЛл][еЕоО0](?=\s*\d)", "№"),
        (r"\bN[o0О°](?=\s*\d)", "№"),
        (r"\b[уУyY][рpP]\b", "р/р"),
        (r"\bШАЗ(?=\d)", "UA"),
        (r"\bУА(?=\d)", "UA"),
        (r"АБ'УКРГАЗБАНК", 'АБ "УКРГАЗБАНК"'),
        (r'[«"]?\s*[кКK][вВB][иИІiI][тТT][аАA][нНH][цЦC][іІiI][яЯR]', 'КВИТАНЦІЯ'),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r'№\s*', '№ ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'(\d{2}\.\d{2}\.\d{4}р?\.)\s*(КВИТАНЦІЯ)', r'\1\n\2', text)
    text = re.sub(r'(КВИТАНЦІЯ)\s*(до платіжної інструкції)', r'\1\n\2', text, flags=re.IGNORECASE)
    return text.strip()


def smart_capitalize_word(word: str) -> str:
    pure = re.sub(r"[^A-Za-zА-Яа-яІіЇїЄєҐґ'-]", "", word)
    if len(pure) <= 1:
        return word
    if pure.isupper() and len(pure) <= 3:
        return word

    upper_count = sum(1 for ch in pure if ch.isupper())
    lower_count = sum(1 for ch in pure if ch.islower())
    suspicious_case = pure.isupper() or (upper_count >= 2 and lower_count >= 1) or (upper_count >= max(2, len(pure) // 2))
    if not suspicious_case:
        return word

    chars = list(word)
    alpha_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if not alpha_positions:
        return word

    first_alpha_idx = alpha_positions[0]
    for idx in alpha_positions:
        chars[idx] = chars[idx].upper() if idx == first_alpha_idx else chars[idx].lower()
    return "".join(chars)


def normalize_line_case(line: str) -> str:
    tokens = re.split(r"(\s+)", line)
    normalized = [smart_capitalize_word(token) if not token.isspace() else token for token in tokens]
    return "".join(normalized)


def normalize_sentence_like_line(line: str) -> str:
    if not line:
        return line

    words = line.split()
    if not words:
        return line

    result_words = []
    first_alpha_word_seen = False
    for word in words:
        chars = list(word)
        alpha_positions = [idx for idx, ch in enumerate(chars) if ch.isalpha()]
        if not alpha_positions:
            result_words.append(word)
            continue

        first_alpha_idx = alpha_positions[0]
        for idx in alpha_positions:
            chars[idx] = chars[idx].upper() if (not first_alpha_word_seen and idx == first_alpha_idx) else chars[idx].lower()
        result_words.append("".join(chars))
        first_alpha_word_seen = True

    return " ".join(result_words)


def title_case_token(token: str) -> str:
    parts = token.split("-")
    fixed_parts = []
    for part in parts:
        chars = list(part)
        alpha_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
        if not alpha_positions:
            fixed_parts.append(part)
            continue

        first_idx = alpha_positions[0]
        for idx in alpha_positions:
            chars[idx] = chars[idx].upper() if idx == first_idx else chars[idx].lower()
        fixed_parts.append("".join(chars))
    return "-".join(fixed_parts)


def looks_like_author_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(ch.isdigit() for ch in stripped):
        return False
    if any(ch in ",:;.!?()" for ch in stripped):
        return False

    words = stripped.split()
    if not (2 <= len(words) <= 4):
        return False

    pure_words = [re.sub(r"[^A-Za-zА-Яа-яІіЇїЄєҐґ'-]", "", w) for w in words]
    if any(len(w) < 2 for w in pure_words):
        return False

    stopwords = {
        "і", "й", "та", "в", "у", "з", "із", "зі", "до", "на", "по", "при",
        "над", "під", "між", "поміж", "для", "про", "від", "без", "через",
    }

    has_valid_hyphen_name = False
    for word in pure_words:
        low = word.lower()
        if low in stopwords:
            return False
        if "-" in word:
            parts = [p for p in word.split("-") if p]
            if len(parts) < 2:
                return False
            if any(part.lower() in stopwords or len(part) < 2 for part in parts):
                return False
            has_valid_hyphen_name = True
    return has_valid_hyphen_name


def normalize_author_line(line: str) -> str:
    return " ".join(title_case_token(word) for word in line.split())


def normalize_eval_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("’", "'").replace("`", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" ?([,:;.!?])", r"\1", text)
    return text.strip().lower()