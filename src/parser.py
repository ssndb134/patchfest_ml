import re
import logging 
from datetime import datetime
from dateutil import parser as date_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#valid months to avoid ambiguous months 
VALID_MONTHS = {
    "jan", "january",
    "feb", "february",
    "mar", "march",
    "apr", "april",
    "may",
    "jun", "june",
    "jul", "july",
    "aug", "august",
    "sep", "sept", "september",
    "oct", "october",
    "nov", "november",
    "dec", "december"
}
#all possible date patterns to be converted
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
    r"\b\d{1,2}\s[A-Za-z]{3,9}\s\d{2,4}\b",
    r"\b[A-Za-z]{3,9}\s\d{1,2},?\s\d{2,4}\b",
    r"\b\d{1,2}\s[A-Za-z]{3,9}\b"
]
#flagging warning for ambi(guous) months
def has_ambi_month (raw_date: str) -> bool: 
    words = re.findall(r"[A-Za-z]{3,9}", raw_date)
    for w in words:
        if w.lower() not in VALID_MONTHS:
            return True
    return False

def extract_date(text: str) -> str | None:
   
    for p in DATE_PATTERNS:
        matches = re.findall(p, text)

        for raw_date in matches:
            try:
                if not re.search(r"\d{4}", raw_date):
                    logger.warning(f"missing year: {raw_date}")

                if has_ambi_month(raw_date):
                    logger.warning(f"ambiguous date: {raw_date}")

                parsed = date_parser.parse(
                    raw_date,
                    dayfirst=True,
                    fuzzy=True,
                    default=datetime(datetime.now().year, 1, 1)
                )

                return parsed.strftime("%Y-%m-%d")

            except Exception:
                continue

    logger.info("no valid date")
    return None

def extract_time(text):
    m=re.search(r"\d{1,2}:\d{2}",text)
    return m.group() if m else None

def parse_ticket(text):
    return {
        "date": extract_date(text),
        "time": extract_time(text)
    }
