import re
import json
import os
from typing import Optional, Dict, Any, List

MOVIE_NAME_PATTERN = r"Movie:?\s*([A-Za-z0-9\s:&'-]+?)\s*(?:Date|Time|Seat|Screen|$)"
DATE_PATTERN = r"(?:Date|Dt):?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
TIME_PATTERN = r"(?:Time|Tm):?\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)"
SEAT_PATTERN = r"(?:Seat|St):?\s*([A-Za-z]{1,3}[-]\d{1,3})"
SCREEN_PATTERN = r"(?:Screen|Scrn)\s*#?\s*(\d+)"

def fallback_date_heuristic(text: str) -> Optional[str]:
    match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}\s*[A-Za-z]{3,}\s*\d{4})", text, re.IGNORECASE)
    return match.group(0).strip() if match else None

def fallback_time_heuristic(text: str) -> Optional[str]:
    match = re.search(r"(\d{2}:\d{2})", text)
    return match.group(0).strip() if match else None

def extract_fields(text: str) -> Dict[str, Any]:
    parsed_data: Dict[str, Optional[str]] = {
        "movie_name": None,
        "date": None,
        "time": None,
        "seat_number": None,
        "screen_number": None,
    }

    extraction_map = {
        "movie_name": (MOVIE_NAME_PATTERN, None),
        "date": (DATE_PATTERN, fallback_date_heuristic),
        "time": (TIME_PATTERN, fallback_time_heuristic),
        "seat_number": (SEAT_PATTERN, None),
        "screen_number": (SCREEN_PATTERN, None),
    }

    for field, (pattern, fallback_func) in extraction_map.items():
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            extracted_value = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
            parsed_data[field] = extracted_value
        elif fallback_func:
            parsed_data[field] = fallback_func(text)

    extracted_count = sum(1 for v in parsed_data.values() if v is not None)
    parsed_data["extraction_status"] = "Success" if extracted_count >= 5 else "Partial Success"

    return parsed_data

def parse_and_save(input_texts: Dict[str, str], output_dir: str = "parsed") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, text in input_texts.items():
        try:
            parsed_output = extract_fields(text)
            output_path = os.path.join(output_dir, f"{filename}.json")
            
            with open(output_path, 'w') as f:
                json.dump(parsed_output, f, indent=4)
                
            print(f"Successfully parsed and saved output for {filename} to {output_path}")

        except Exception as e:
            error_output = {"error": f"Failed to process text: {e}", "original_text_sample": text[:100] + "..."}
            output_path = os.path.join(output_dir, f"{filename}_error.json")
            with open(output_path, 'w') as f:
                json.dump(error_output, f, indent=4)
            print(f"ERROR processing {filename}. Error details saved to {output_path}")

if __name__ == "__main__":
    sample_texts: Dict[str, str] = {
        "ticket_1_complete": (
            "CINEMA Receipt\n"
            "Movie: THE DARK KNIGHT RISES\n"
            "Date: 12/16/2025 Time: 09:45 PM\n"
            "Screen # 1, Seat: J-22"
        ),
        "ticket_2_fallback_needed": (
            "Ticket for\n"
            "Movie: INTERSTELLAR\n"
            "Date: 2025-12-16 Time: 20:30\n"
            "Screen 5, St: A-01"
        ),
        "ticket_3_incomplete": (
            "Random Receipt Data\n"
            "Movie: SHREK 2\n"
            "16/12/25"
        ),
        "ticket_4_error_test": 
            "Movie: THE MATRIX\n" + chr(0xED) 
    }

    parse_and_save(sample_texts)