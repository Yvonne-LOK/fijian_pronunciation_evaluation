# 检查发音词典中是否包含声学模型中不支持的音素
# 如果包含，打印警告信息

import json

with open("/mnt/d/workdir/data/fijian/valid_phones_en.json", "r", encoding="utf-8") as f:
    valid_phones = set(json.load(f)["Phones"])

with open("/mnt/d/workdir/data/fijian/fijian_bootstrap_arpa.dict", "r", encoding="utf-8") as f:
    dict_lines = f.read().splitlines()

invalid_phones = set()
for line in dict_lines:
    phones = line.split()[1:]  # 假设第一个元素是单词，后面的元素是音素
    for phone in phones:
        if phone not in valid_phones:
            invalid_phones.add(phone)

if invalid_phones:
    print("警告：以下音素在声学模型中不被支持：")
    for phone in invalid_phones:
        print(f"  {phone}")
else:
    print("所有音素都在声学模型中被支持。")