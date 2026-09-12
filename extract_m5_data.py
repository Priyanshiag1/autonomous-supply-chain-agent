import urllib.request
import csv
import os
import sys

URL = "https://huggingface.co/datasets/denephew/M5_Forecasting/resolve/main/sales_train_validation.csv"
OUTPUT_FILE = "data_m5_daily_365.csv"

# Columns to extract: metadata columns + d_1 to d_365
DAYS_LIMIT = 365
target_day_cols = [f"d_{i}" for i in range(1, DAYS_LIMIT + 1)]

print(f"Connecting to official Walmart M5 stream: {URL}")
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})

# We want representative SKUs across all 3 states: CA, TX, WI
# We will extract 3 stores: CA_1, TX_1, WI_1 (or all stores for Top categories)
# Let's extract all 3,049 products across CA_1, TX_1, WI_1 (total ~9,147 series)
# which gives exact state-by-state comparisons for the same items!
TARGET_STORES = {"CA_1", "TX_1", "WI_1"}

rows_written = 0
state_counts = {"CA": 0, "TX": 0, "WI": 0}

with urllib.request.urlopen(req, timeout=30) as resp:
    # Read header
    header_line = resp.readline().decode("utf-8").strip()
    headers = list(csv.reader([header_line]))[0]
    
    # Identify indices
    id_idx = headers.index("id")
    item_idx = headers.index("item_id")
    dept_idx = headers.index("dept_id")
    cat_idx = headers.index("cat_id")
    store_idx = headers.index("store_id")
    state_idx = headers.index("state_id")
    
    day_indices = [headers.index(col) for col in target_day_cols]
    
    out_headers = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + target_day_cols
    
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(out_headers)
        
        # Read lines in chunks
        reader = csv.reader((line.decode("utf-8") for line in resp))
        for row in reader:
            if not row:
                continue
            store_id = row[store_idx]
            state_id = row[state_idx]
            
            # Select target stores (CA_1, TX_1, WI_1)
            if store_id in TARGET_STORES:
                meta = [row[id_idx], row[item_idx], row[dept_idx], row[cat_idx], row[store_idx], row[state_idx]]
                days_data = [row[idx] for idx in day_indices]
                writer.writerow(meta + days_data)
                rows_written += 1
                state_counts[state_id] = state_counts.get(state_id, 0) + 1
                
                if rows_written % 1500 == 0:
                    print(f"Extracted {rows_written} time-series records...")

print(f"\nSuccessfully created {OUTPUT_FILE}!")
print(f"Total series written: {rows_written}")
print(f"State breakdown: {state_counts}")
print(f"File size on disk: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
