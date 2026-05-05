import os
root = "Dataset/EmoWork_v2/EmoWorker_v2/CSV/SENSORS_csv"
sids = sorted([d for d in os.listdir(root) if d.isdigit()], key=int)
print("subjects:", len(sids), sids)
for sid in sids[:2] + sids[-2:]:
    for s in ["b1", "b2", "b3", "c1", "c2", "c3"]:
        d = f"{root}/{sid}/{s}"
        if os.path.isdir(d):
            print(sid, s, sorted(os.listdir(d)))
