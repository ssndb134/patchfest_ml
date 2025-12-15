import os
from dateutil import parser
from datetime import datetime
from typing import Optional, List

ISO_FORMAT = "%Y-%m-%d"

def parse_date_to_iso(date_string: str, dayfirst: bool = True) -> Optional[str]:
    if not date_string or not isinstance(date_string, str):
        print("LOG: Input is empty or not a string. Returning None.")
        return None
    try:
        dt_object = parser.parse(date_string, dayfirst=dayfirst)
        return dt_object.strftime(ISO_FORMAT)
    except ValueError as e:
        print(f"LOG: Failed to parse '{date_string}'. Reason: {e}")
        return None
    except Exception as e:
        print(f"LOG: Unexpected error during parsing '{date_string}': {e}")
        return None

def run_parser_test(samples: List[str]):
    print("--- Testing Universal Date Parser ---")
    for sample in samples:
        iso_date = parse_date_to_iso(sample)
        status = "SUCCESS" if iso_date else "FAILURE"
        print(f"Original: '{sample}' -> ISO 8601: '{iso_date}' [{status}]")

if __name__ == '__main__':
    test_dates = [
        "15 Aug 24",
        "2024-08-15",
        "15/08/2024",
        "08.15.2024",
        "August 15th 2024",
        "15-08-24",
        "05/08/24",
        "not a date string",
        "",
    ]
    run_parser_test(test_dates)
