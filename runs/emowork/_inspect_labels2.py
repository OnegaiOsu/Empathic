import os
root = "Dataset/EmoWork_v2/EmoWorker_v2/CSV/LABEL_csv"
for sid in ["1", "2", "3", "5", "10", "30"]:
    p = os.path.join(root, sid)
    print(sid, sorted(os.listdir(p)) if os.path.isdir(p) else "X")
