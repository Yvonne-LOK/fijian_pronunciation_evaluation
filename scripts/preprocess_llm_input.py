import json

MAX_FOCUS_POINTS = None  #2

def convert_raw_to_llm(raw_json):

    sentence_text = raw_json["reference_text"]
    sentence_score = raw_json["sentence"]["score_relative"]

    focus_words = []

    for w in raw_json.get("weak_words", []):

        word_entry = {
            "word": w["word"],
            "score": round(w["word_score_relative"], 2)
        }

        # 处理音素问题
        weak_phones = w.get("weak_phones", [])
        if weak_phones:
            phones = [{p["phone"]: round(p["phone_score_relative"], 2)} for p in weak_phones]
            word_entry["focus_phones"] = phones

        focus_words.append(word_entry)

    # 按score排序
    focus_words = sorted(focus_words, key=lambda x: x["score"])

    # 截断
    if MAX_FOCUS_POINTS is not None:    
        focus_words = focus_words[:MAX_FOCUS_POINTS]

    llm_json = {
        "sentence_text": sentence_text,
        "sentence_score": round(sentence_score, 2),
        "focus_words": focus_words,
        # "feedback_policy": {
        #     "max_focus_points": MAX_FOCUS_POINTS
        # }
    }

    return llm_json


if __name__ == "__main__":

    # with open("raw_result.json", "r", encoding="utf-8") as f:
    #     raw = json.load(f)
    
    with open("/mnt/d/workdir/outputs/eval_runs/learner_guoziyu_20260312_012224/llm_diagnosis_input.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            raw: dict = json.loads(line)
            print(f"{raw['reference_text']}: {raw['sentence']['score_relative']}")
    exit(0)
    
    # 示例：读指定一行的dict
    with open("/mnt/d/workdir/outputs/eval_runs/learner_guoziyu_20260312_012224/llm_diagnosis_input.jsonl", "r", encoding="utf-8") as f:
        raw: dict = json.loads(f.readlines()[0])

    llm_input = convert_raw_to_llm(raw)

    print(json.dumps(llm_input, ensure_ascii=False, indent=2))