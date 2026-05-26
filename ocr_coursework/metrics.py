from typing import Any, Dict, Sequence
from .text_utils import normalize_eval_text


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    len1 = len(seq1)
    len2 = len(seq2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len2]


def evaluate_ocr_against_ground_truth(recognized_text: str, ground_truth_text: str) -> Dict[str, Any]:
    rec_norm = normalize_eval_text(recognized_text)
    gt_norm = normalize_eval_text(ground_truth_text)

    char_dist = levenshtein_distance(rec_norm, gt_norm)
    gt_char_len = max(len(gt_norm), 1)
    cer = char_dist / gt_char_len
    char_accuracy = max(0.0, 1.0 - cer)

    rec_words = rec_norm.split()
    gt_words = gt_norm.split()
    word_dist = levenshtein_distance(rec_words, gt_words)
    gt_word_len = max(len(gt_words), 1)
    wer = word_dist / gt_word_len
    word_accuracy = max(0.0, 1.0 - wer)

    return {
        "ground_truth_chars": len(gt_norm),
        "recognized_chars": len(rec_norm),
        "ground_truth_words": len(gt_words),
        "recognized_words": len(rec_words),
        "char_edit_distance": int(char_dist),
        "word_edit_distance": int(word_dist),
        "cer": round(cer, 6),
        "char_accuracy": round(char_accuracy, 6),
        "wer": round(wer, 6),
        "word_accuracy": round(word_accuracy, 6),
    }