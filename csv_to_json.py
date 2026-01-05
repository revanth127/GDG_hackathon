import csv
import json

# csv_file = "laps.csv"
# json_file = "laps.json"

# with open(csv_file, newline='', encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     rows = list(reader)

# with open(json_file, "w", encoding="utf-8") as f:
#     json.dump(rows, f, indent=2)

# print("Converted CSV to JSON successfully")

import json

with open("laps.json", "r", encoding="utf-8") as f:
    raw_laps = json.load(f)

def to_float(x):
    try:
        return float(x)
    except:
        return None

def clean_lap(lap):
    return {
        "driver": lap.get("Driver"),
        "driver_number": int(float(lap["DriverNumber"])) if lap.get("DriverNumber") else None,
        "lap": int(float(lap["LapNumber"])) if lap.get("LapNumber") else None,
        "lap_time": to_float(lap.get("LapTime_in_seconds")),
        "compound": lap.get("Compound"),
        "tyre_life": int(float(lap["TyreLife"])) if lap.get("TyreLife") else None,
        "fresh_tyre": lap.get("FreshTyre") == "True",
        "position": int(float(lap["Position"])) if lap.get("Position") else None,
        "sector_times": [
            to_float(lap.get("Sector1Time")),
            to_float(lap.get("Sector2Time")),
            to_float(lap.get("Sector3Time"))
        ],
        "speed_trap": to_float(lap.get("SpeedST")),
        "team": lap.get("Team"),
        "track_status": "GREEN" if lap.get("TrackStatus") == "1" else "OTHER"
    }

cleaned_laps = [clean_lap(l) for l in raw_laps]

with open("laps_clean.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_laps, f, indent=2)

print("Clean JSON saved to laps_clean.json")

