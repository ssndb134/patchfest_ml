import json
import logging
import re
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


def _safe_match(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group().strip()
    return None


def extract_date(text):
    # Regex patterns cover ISO, slash, dot, and textual month formats
    patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4}\b",
    ]
    date_str = _safe_match(patterns, text)
    if date_str:
        return date_str

    # Fallback: try to parse tokens that look like dates
    for token in text.split():
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(token, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def extract_time(text):
    patterns = [
        r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b",
        r"\b\d{1,2}\s?(?:AM|PM|am|pm)\b",
        r"\b\d{3,4}\s?(?:AM|PM|am|pm)\b",  # compact times like 0730PM
    ]
    return _safe_match(patterns, text)


def extract_seat(text):
    patterns = [
        r"\b(?:Seat|Row)?\s*[A-Z]{1,3}\d{1,3}\b",
        r"\b[A-Z]{1,3}-?\d{1,3}\b",
    ]
    return _safe_match(patterns, text)


def extract_screen(text):
    patterns = [r"\b(?:Screen|Audi|Auditorium)\s*\d+\b"]
    return _safe_match(patterns, text)


def extract_movie_name(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    ignore_patterns = [
        r"\b(Screen|Audi|Auditorium)\s*\d+\b",
        r"\bSeat\b|\bRow\b",
        r"\d{1,2}:\d{2}",
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
    ]

    def is_meta(line):
        return any(re.search(p, line, flags=re.IGNORECASE) for p in ignore_patterns)

    def is_numeric_only(line):
        return line.replace(" ", "").isdigit()

    candidates = [ln for ln in lines if not is_meta(ln) and not is_numeric_only(ln) and len(ln) > 3]
    if candidates:
        candidates.sort(key=lambda s: (-len(s), s))
        return candidates[0]
    return lines[0] if lines else None


def parse_ticket(text):
    text = text or ""
    try:
        parsed = {
            "movie_name": extract_movie_name(text),
            def _normalize_date(date_str):
                """Normalize a date string to ISO-8601 (YYYY-MM-DD) when possible."""
                date_str = date_str.strip()

                # Ambiguity check for purely numeric day/month swaps
                mm_dd_candidates = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$", date_str)
                if mm_dd_candidates:
                    a, b, year_part = mm_dd_candidates.groups()
                    if int(a) <= 12 and int(b) <= 12:
                        logger.warning("Ambiguous month/day detected in '%s'", date_str)

                # Formats with explicit year
                formats_with_year = [
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%d-%m-%Y",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                    "%m-%d-%Y",
                    "%d %b %Y",
                    "%d %B %Y",
                    "%d %b %y",
                    "%d %B %y",
                ]

                for fmt in formats_with_year:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.date().isoformat()
                    except ValueError:
                        continue

                # Formats missing year: log and return None
                formats_missing_year = ["%d %b", "%d %B"]
                for fmt in formats_missing_year:
                    try:
                        datetime.strptime(date_str, fmt)
                        logger.warning("Missing year in date '%s'", date_str)
                        return None
                    except ValueError:
                        continue

                return None


            def extract_date(text):
                """Extract and normalize dates to ISO-8601."""
                if not text:
                    return None

                patterns = [
                    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
                    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
                    r"\b\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4}\b",
                    r"\b\d{1,2}\s?[A-Za-z]{3,9}\b",  # missing year
                ]

                # Collect candidates to try normalization
                candidates = []
                for pattern in patterns:
                    match = re.search(pattern, text, flags=re.IGNORECASE)
                    if match:
                        candidates.append(match.group())

                # Fallback: token-wise search
                if not candidates:
                    candidates.extend(text.split())

                for candidate in candidates:
                    normalized = _normalize_date(candidate)
                    if normalized:
                        return normalized

                # No valid date found; log and return None without raising
                logger.info("No parseable date found in text snippet")
                return None
    parsed = parse_ticket(text)
    ident = identifier or "ticket"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{ident}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return str(file_path)
