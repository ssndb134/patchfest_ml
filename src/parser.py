import re
import json
import os
from datetime import datetime

def clean_text(text):
    """Clean extra whitespace from text."""
    return text.strip()

def extract_date(text):
    """Extract date using regex patterns with fallbacks."""
    patterns = [
        r"(\d{4}-\d{2}-\d{2})",                  # YYYY-MM-DD
        r"(\d{2}/\d{2}/\d{4})",                  # DD/MM/YYYY
        r"(\d{2}-\d{2}-\d{4})",                  # DD-MM-YYYY
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})", # 12 Dec 2022
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})"  # Dec 12, 2022
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    
    #  Look for "Date" keyword near something looking like a date (digits)
    m = re.search(r"Date\s*:?\s*([A-Za-z0-9\s,\/\-]+)", text, re.IGNORECASE)
    if m:
        #  return the captured group if it has numbers
        candidate = m.group(1).strip()
        if any(c.isdigit() for c in candidate) and len(candidate) < 20: 
             return candidate
    return None

def extract_time(text):
    """Extract time using regex patterns with fallbacks."""
    patterns = [
        r"(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm))",   # 07:00 PM
        r"(\d{1,2}:\d{2})"                      # 14:00
    ]
    for p in patterns:
        m = re.search(p, text)
        if m: return m.group(1)
    
    #  Look for "Time" keyword
    m = re.search(r"Time\s*:?\s*(\d{1,2}\s?:\s?\d{2}.*)", text, re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def extract_screen(text):
    """Extract screen number or name."""
    patterns = [
        r"(?:Screen|Scrn|Audi|Hall)\s*:?\s*([A-Z0-9]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None

def extract_seat(text):
    """Extract seat number using various formats."""
    patterns = [
        r"Row\s*([A-Z0-9]+)\s*Seat\s*([A-Z0-9]+)", # Row A Seat 12 - Prioritize this!
        r"Seat\s*:?\s*([A-Z]?\s?-?\s?\d+)",       # Seat: A12, Seat 43
        r"(?:Seat|St)\s*([A-Z]-[0-9]+)"            # Seat A-12
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if len(m.groups()) == 2: # Case with Row and Seat groups
                return f"{m.group(1)}-{m.group(2)}"
            return m.group(1).replace(" ", "")
    
    #  look for standalone patterns like "G 43" if preceded by relevant text?
    # sticking to keyword-based for now to reduce FPS.
    return None

def extract_movie_name(text):
    """extraction of movie name."""
    # Split lines
    lines = [clean_text(Line) for Line in text.split('\n') if clean_text(Line)]
    
    # Common words in tickets to ignore when guessing title
    ignore_keywords = [
        "TICKET", "CINEMA", "THEATRE", "PVR", "INOX", "CINEPOLIS", "SHOW", 
        "DATE", "TIME", "SCREEN", "AUDI", "SEAT", "ROW", "PRICE", "TOTAL", 
        "AMOUNT", "TAX", "ID", "BOOKING", "RS.", "INR", "PAYMENT", "MOVIE",
        "CONFIRMATION"
    ]
    
    candidates = []
    for line in lines:

        # 1. Skip lines with dates or times
        if extract_date(line) or extract_time(line):
            continue
            
        # 2. Skip lines purely numbers or prices
        if re.match(r'^[\d\s\.,]+$', line):
            continue
            
        # 3. Check for Ignore Keywords
        upper_line = line.upper()
        if any(keyword in upper_line for keyword in ignore_keywords):
            # If the line contains a keyword
            # BUT: Check if the keyword is just a prefix? e.g. "Movie: Spiderman"
            # If "Movie" is in line, we extract what's AFTER it.
            
            # Check for specific prefixes
            for prefix in ["MOVIE", "SHOW"]:
                if upper_line.startswith(prefix):
                    # Remove prefix and special chars
                    cleaned = re.sub(rf"^{prefix}[:\s]*", "", line, flags=re.IGNORECASE).strip()
                    if cleaned and len(cleaned) > 2:
                        candidates.append(cleaned)
                    # Don't continue
                    break 
            
            # skip generic keyword line
            if "TICKET" in upper_line or "CINEMA" in upper_line or "PRICE" in upper_line:
                continue
            pass 
        
        # 4. All Caps or significant length logic
        candidates.append(line)

    # Return the first candidate that isn't a known generic keyword.
    
    for c in candidates:
        if len(c) < 3: continue
        # strict equality with keywords
        if c.upper() in ignore_keywords: continue
        return c

    return None

def parse_ticket(text):
    
    if not text or not isinstance(text, str):
        return {}
    
    # Extract fields
    try:
        movie = extract_movie_name(text)
        date_val = extract_date(text)
        time_val = extract_time(text)
        screen = extract_screen(text)
        seat = extract_seat(text)
        
        return {
            "movie_name": movie,
            "date": date_val,
            "time": time_val,
            "screen_number": screen,
            "seat_number": seat,
        }
    except Exception as e:
        print(f"Extraction error: {e}")
        return {} # Fail gracefully

def save_parsed_data(data, output_dir="parsed"):
    """Saves the parsed data dictionary to a JSON file."""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            pass 
            
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_part = data.get("movie_name", "unknown")
    if name_part:
        name_part = re.sub(r'[^\w\-_]', '', name_part)[:15]
    else:
        name_part = "ticket"
        
    filename = f"{name_part}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to save data: {e}")
        return None

if __name__ == "__main__":
    # Test with sample noisy data
    samples = [
        '''
        PVR CINEMAS
        AVATAR: THE WAY OF WATER
        Date: 14 Dec 2022
        Time: 07:00 PM
        Screen 4  Seat: G 12
        Price: 240.00
        ''',
        '''
        BATMAN BEGINS
        12-05-2023 18:30
        AUDI 02
        ROW C SEAT 4
        ''',
        '''
        Super Cinema
        Ticket Confirmation
        Movie: SPIDERMAN NO WAY HOME
        Fri, 12 Oct 2023
        10:00 AM
        Scrn 1 Seat A-2
        '''
    ]
    
    for i, s in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        parsed = parse_ticket(s)
        print(parsed)
        save_parsed_data(parsed, "parsed")
