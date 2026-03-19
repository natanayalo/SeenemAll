import json
from pathlib import Path
from collections import Counter

EVAL_PATH = Path("evaluation/evaluation_set_v2.json")
OUTPUT_PATH = Path("evaluation/evaluation_set_v2.json")

def deduplicate_and_rebalance():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Deduplicate based on golden_set (Jaccard 1.0)
    seen_golden_sets = {}
    unique_data = []
    
    # We want to keep the "v2_case_*" IDs if possible as they are the primary set
    # Sort data to prioritize v2_case_*
    data.sort(key=lambda x: (not x.get("case_id", "").startswith("v2_case"), x.get("case_id", "")))

    for entry in data:
        golden_set = entry.get("golden_set", [])
        # Convert golden set to a tuple of sorted IDs for hashability
        golden_key = tuple(sorted([item["id"] for item in golden_set]))
        
        if golden_key not in seen_golden_sets:
            seen_golden_sets[golden_key] = entry["case_id"]
            unique_data.append(entry)
        else:
            print(f"Skipping duplicate: {entry['case_id']} (Duplicate of {seen_golden_sets[golden_key]})")

    # 2. Rebalance high-frequency labels
    # 96677: Lupin, 71738: The Orville
    # We'll remove these from cases where they are "out of place" or just to reduce frequency
    
    count_96677 = 0
    count_71738 = 0
    
    for entry in unique_data:
        case_id = entry.get("case_id", "")
        bucket = entry.get("distribution_bucket", "")
        
        # Remove Lupin (96677) from Non-Crime cases
        if "Crime" not in (entry.get("genre_override", "") or "") and bucket not in ["semantic_query_tv", "constrained_multi_filter"]:
            if any(item["id"] == 96677 for item in entry["golden_set"]):
                entry["golden_set"] = [item for item in entry["golden_set"] if item["id"] != 96677]
                print(f"Removed 96677 (Lupin) from {case_id}")

        # Remove The Orville (71738) if it appears too much in generic sci-fi
        if any(item["id"] == 71738 for item in entry["golden_set"]):
            count_71738 += 1
            if count_71738 > 8 and bucket == "semantic_query_tv":
                entry["golden_set"] = [item for item in entry["golden_set"] if item["id"] != 71738]
                print(f"Reduced frequency: Removed 71738 from {case_id}")

    # 3. Backfill weak labels (golden_set < 4)
    for entry in unique_data:
        if len(entry["golden_set"]) < 4:
            case_id = entry.get("case_id", "")
            if "psychological thrillers" in (entry.get("query") or "").lower():
                entry["golden_set"].append({"id": 106646, "title": "The Wolf of Wall Street", "media_type": "movie"})
            elif "post-apocalyptic" in (entry.get("query") or "").lower():
                entry["golden_set"].append({"id": 204541, "title": "Silo", "media_type": "tv"})
            elif "superhero" in (entry.get("query") or "").lower():
                entry["golden_set"].append({"id": 95557, "title": "Invincible", "media_type": "tv"})
            elif "family adventure" in (entry.get("query") or "").lower():
                 entry["golden_set"].append({"id": 11, "title": "Star Wars", "media_type": "movie"})

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=2)
    
    print(f"Final case count: {len(unique_data)}")

if __name__ == "__main__":
    deduplicate_and_rebalance()
