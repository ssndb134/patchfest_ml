import pandas as pd
import numpy as np
import os
import re

RAW_DATA_DIR = 'data/raw'
INTERIM_DATA_PATH = 'data/interim/ocr_merged.csv'

def simulate_parse_ticket(raw_text: str):
    movie_match = re.search(r"Movie:\s*([A-Za-z0-9\s]+)", raw_text)
    date_match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", raw_text)
    seat_match = re.search(r"Seat:\s*([A-Z]{1}-\d{2})", raw_text)
    time = None if 'UNREADABLE_TIME' in raw_text else "20:00"
    return {
        "movie_name": movie_match.group(1).strip() if movie_match else None,
        "date": date_match.group(1) if date_match else None,
        "time": time,
        "seat_number": seat_match.group(1) if seat_match else None,
        "screen_number": np.random.randint(1, 10),
        "raw_text_length": len(raw_text)
    }

def simulate_impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df['time'].fillna('18:00', inplace=True)
    df['movie_name'].fillna('UNKNOWN', inplace=True)
    df['screen_number'].fillna(df['screen_number'].median(), inplace=True)
    return df

def run_ocr_pipeline_stage():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    raw_tickets = {
        "ticket_001.png": "Receipt Movie: INTERSTELLAR Date: 2025-12-16 Seat: A-01 Screen: 5",
        "ticket_002.png": "Movie: BARBIE Date: 2025-07-25 Seat: C-12 Screen: 2",
        "ticket_003.png": "UNREADABLE_IMAGE_FLAG",
        "ticket_004.png": "Movie: OPPENHEIMER Date: 2025-07-25 Seat: B-05 Screen: 2 UNREADABLE_TIME",
        "ticket_005.png": "Movie: DUNE Date: 2025-08-01 Seat: F-07 Screen: 8",
    }
    for filename, text in raw_tickets.items():
        with open(os.path.join(RAW_DATA_DIR, filename), 'w') as f:
            f.write(text)
    all_parsed_data = []
    print("Starting OCR -> Parsing Pipeline...")
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        try:
            with open(file_path, 'r') as f:
                raw_text = f.read()
            if "UNREADABLE_IMAGE_FLAG" in raw_text:
                print(f"Skipping {filename}: Image unreadable.")
                continue
            parsed_json = simulate_parse_ticket(raw_text)
            parsed_json['source_file'] = filename
            all_parsed_data.append(parsed_json)
            print(f"Processed {filename}.")
        except Exception as e:
            print(f"ERROR: Failed to process {filename}. Error: {e}")
    if not all_parsed_data:
        print("No data processed. Exiting pipeline stage.")
        return
    df_raw = pd.DataFrame(all_parsed_data)
    df_interim = simulate_impute_missing(df_raw)
    os.makedirs(os.path.dirname(INTERIM_DATA_PATH), exist_ok=True)
    df_interim.to_csv(INTERIM_DATA_PATH, index=False)
    print(f"\nPipeline stage complete. Saved aggregated data ({len(df_interim)} rows) to {INTERIM_DATA_PATH}")

if __name__ == '__main__':
    run_ocr_pipeline_stage()
