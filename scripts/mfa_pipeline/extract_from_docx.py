from docx import Document
import re


def clean_string(s):
    """
    清洗字符串：去除中文字符、所有标点、首尾空格，字母转小写
    :param s: 原始字符串（如 'Bula. 你好。'）
    :return: 清洗后的纯小写英文字符串（如 'bula'）
    """
    # 1. 去除所有中文字符（匹配[\u4e00-\u9fa5]范围内的所有汉字并替换为空）
    no_chinese = re.sub(r'[\u4e00-\u9fa5]', '', s)
    # 2. 去除所有标点符号（匹配非字母数字的字符并替换为空）
    no_punctuation = re.sub(r'[^\w\s]', '', no_chinese)
    # 3. 去除首尾空格 + 字母转小写
    cleaned = no_punctuation.replace("   ", " ").strip().lower()
    
    return cleaned

lesson = "Wase_Dua-1"
doc = Document(f"/mnt/d/workdir/data/fijian/text_book/{lesson}.docx")

print(doc.paragraphs)  
print("tables:", len(doc.tables))

# 遍历word文件里面表格内容
final_lines = []
for ti, table in enumerate(doc.tables):
    print(f"\n=== TABLE {ti} ===")
    for ri, row in enumerate(table.rows):
        row_texts = []
        for ci, cell in enumerate(row.cells):
            text = cell.text.strip().replace("\n", " | ")
            clean_text = clean_string(text)
            if clean_text:  # 只添加非空的清洗后文本    
                row_texts.append(clean_text)
        # print("\n".join(row_texts), "\n")
        # import ipdb; ipdb.set_trace()
        final_lines.extend(row_texts)

print("\n=== ALL LINES ===")
for line in final_lines[:121]:
    print(line) 
    
print(f"\nTotal liness in {lesson}: {len(final_lines)}")

with open(f"/mnt/d/workdir/data/fijian/text_book/cleaned_{lesson}.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(final_lines))
print(f"\n已导出 cleaned_{lesson}.txt")

# for ti, table in enumerate(doc.tables):
#     print(f"\n=== TABLE {ti} ===")
#     for ri, row in enumerate(table.rows):
#         row_texts = []
#         for ci, cell in enumerate(row.cells):
#             text = cell.text.strip().replace("\n", " | ")
#             row_texts.append(f"[{ri},{ci}] {repr(text)}")
#         import ipdb; ipdb.set_trace()
#         print(" ; ".join(row_texts))