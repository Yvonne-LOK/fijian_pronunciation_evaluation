import os
import re

corpus_dir = "/mnt/d/workdir/data/fijian/raw_corpus"
lexicon = {}

for file in os.listdir(corpus_dir):
    if file.endswith(".txt"):
        with open(os.path.join(corpus_dir, file), "r", encoding="utf-8") as f:
            text = f.read().lower()

        words = re.findall(r"[a-zāēīōū]+", text)

        for w in words:
            if w not in lexicon:
                lexicon[w] = " ".join(list(w))

dict_path = "/mnt/d/workdir/data/fijian/fijian.dict"
with open(dict_path, "w", encoding="utf-8") as f:
    for w, p in sorted(lexicon.items()):
        f.write(f"{w} {p}\n")

print("Lexicon size:", len(lexicon))
print(f"已导出至 {dict_path}")