

with open("/mnt/d/workdir/data/fijian/text_book/cleaned_Wase_Dua-1_20.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
    
prefix = "u1_l1"    
    
for i in range(len(lines)):
    with open(f"/mnt/d/workdir/data/fijian/raw_corpus/{prefix}_{i+1:03d}.txt", "w", encoding="utf-8") as f:
        f.write(lines[i])