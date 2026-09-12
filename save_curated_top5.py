import pandas as pd
import numpy as np

df = pd.read_csv("data_m5_daily_365.csv")
cal = pd.read_csv("m5_repo/calendar.csv")

day_cols = [f"d_{i}" for i in range(1, 366)]

# Curated list of 5 Hero SKUs covering:
# 1. SuperBowl Sunday Hero (FOODS_3_090_TX_1)
# 2. Mega Outlier SNAP Hero (FOODS_2_285_TX_1)
# 3. California High-Velocity Hero (FOODS_3_030_CA_1)
# 4. Hobbies / Gift Category Hero (HOBBIES_1_209_TX_1)
# 5. Household Cleaning / Goods Hero (HOUSEHOLD_2_440_TX_1)

target_ids = [
    "FOODS_3_090_TX_1_validation",
    "FOODS_2_285_TX_1_validation",
    "FOODS_3_030_CA_1_validation",
    "HOBBIES_1_209_TX_1_validation",
    "HOUSEHOLD_2_440_TX_1_validation"
]

rows = []
for tid in target_ids:
    r = df[df['id'] == tid].iloc[0]
    vals = r[day_cols].values.astype(float)
    mean_val = np.mean(vals)
    std_val = np.std(vals, ddof=1)
    cv_val = std_val / mean_val
    max_val = np.max(vals)
    max_day = day_cols[np.argmax(vals)]
    peak_ratio = max_val / mean_val
    
    # Check calendar metadata
    cal_row = cal[cal['d'] == max_day].iloc[0]
    evt = str(cal_row['event_name_1']) if pd.notna(cal_row['event_name_1']) else "None"
    snap_flag = cal_row[f"snap_{r['state_id']}"]
    
    rows.append({
        'id': r['id'],
        'item_id': r['item_id'],
        'dept_id': r['dept_id'],
        'cat_id': r['cat_id'],
        'store_id': r['store_id'],
        'state_id': r['state_id'],
        'mean_sales': round(mean_val, 2),
        'std_sales': round(std_val, 2),
        'cv': round(cv_val, 2),
        'max_sales': int(max_val),
        'max_day': max_day,
        'max_date': cal_row['date'],
        'weekday': cal_row['weekday'],
        'peak_ratio': round(peak_ratio, 1),
        'calendar_event': evt,
        'snap_flag': snap_flag
    })

curated_df = pd.DataFrame(rows)
curated_df.to_csv("top_5_demo_skus.csv", index=False)

print("="*75)
print("             CURATED TOP 5 HERO DEMO SKUS")
print("="*75)
for idx, r in curated_df.iterrows():
    print(f"\n[{idx+1}] {r['id']}")
    print(f"    Category: {r['cat_id']} | State: {r['state_id']} | Store: {r['store_id']}")
    print(f"    Baseline Mean: {r['mean_sales']} units/day | Std Dev: {r['std_sales']} | CV: {r['cv']}")
    print(f"    Dramatic Peak: {r['max_sales']} units on {r['max_day']} ({r['max_date']}, {r['weekday']}) -> {r['peak_ratio']}x baseline!")
    print(f"    Calendar Driver: Event='{r['calendar_event']}', SNAP_Welfare_Flag={r['snap_flag']}")
print("="*75)
