import pandas as pd
from collections import Counter

# ===== 4-model vote =====
# serverGetPoint：直接用 seed3407
# actionId / pointId：3407 + 3408 + 777 + base 投票
# 平手優先順序：3407 > 3408 > 777 > base

file_seed3407 = "submission_h256_k5_seed3407.csv"
file_seed3408 = "submission_h256_k5_seed3408.csv"
file_seed777  = "submission_h256_k5_seed777.csv"
file_base     = "submission_h256_k5.csv"

out_file = "submission_vote_3407_3408_777_base.csv"


def vote_with_priority(vals):
    cnt = Counter(vals)
    max_count = max(cnt.values())
    candidates = {v for v, c in cnt.items() if c == max_count}

    # 平手時照順序優先
    for v in vals:
        if v in candidates:
            return v

    return vals[0]


s3407 = pd.read_csv(file_seed3407)
s3408 = pd.read_csv(file_seed3408)
s777 = pd.read_csv(file_seed777)
sbase = pd.read_csv(file_base)

# 以 seed3407 的 rally_uid 為主
base_uid = s3407["rally_uid"]


def align(df):
    if df["rally_uid"].equals(base_uid):
        return df.copy()
    return df.set_index("rally_uid").loc[base_uid].reset_index()


s3408 = align(s3408)
s777 = align(s777)
sbase = align(sbase)

# 建立輸出
out = s3407.copy()

# serverGetPoint：直接用 seed3407
out["serverGetPoint"] = s3407["serverGetPoint"]

# actionId：4-model vote
out["actionId"] = [
    vote_with_priority([a, b, c, d])
    for a, b, c, d in zip(
        s3407["actionId"],
        s3408["actionId"],
        s777["actionId"],
        sbase["actionId"],
    )
]

# pointId：4-model vote
out["pointId"] = [
    vote_with_priority([a, b, c, d])
    for a, b, c, d in zip(
        s3407["pointId"],
        s3408["pointId"],
        s777["pointId"],
        sbase["pointId"],
    )
]

# 保持欄位順序
out = out[s3407.columns]

# 輸出 csv
out.to_csv(out_file, index=False)

print("Saved:", out_file)
print("Rows:", len(out))
print(out.head())
