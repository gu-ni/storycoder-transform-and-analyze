# %%
import os
import json, re, math
from collections import Counter

def extract_core_algorithm(text):
    if not text:
        return ""
    m = re.search(r"Core Algorithm:\s*(.+)", text)
    return m.group(1).strip() if m else ""

def extract_algo_category(text):
    m = re.search(r"Algorithm Category:\s*(.+)", text)
    return m.group(1).strip() if m else ""


# =======================
# 실행 옵션
# =======================
backend = "gemini"                        # ["gemini", "chatgpt", "claude"]
benchmark = "codeforces_both"                       # ["humaneval_filtered", "test6", "codeforces", "codeforces_challenging", "codeforces_both"]
compare_mode = "both"                     # "with_io", "merge", "both"
gold_selection_mode = "majority"          # "unique" or "majority"
analysis_mode = "passk_prop"    # "matching", "passk", "passk_soft", "passk_prop"
k_list = [10]                       # k 값들

# 새로운 플래그
wrong_threshold = 1.0                      # match+wrong vs mismatch+wrong 기준
soft_threshold = 0.1                       # match+correct 판정 기준 (passk_soft 모드에서 사용)
exclude_both_incorrect = False             # original & narrative 둘 다 틀리면 제외


# =======================
# 데이터 경로
# =======================
benchmark_dict = {
    "humaneval_filtered": "HumanEval",
    "test6": "LiveCodeBench",
    "codeforces": "CodeForces",
    "codeforces_challenging": "CodeForces",
    "codeforces_both": "CodeForces",  # placeholder
}

base_dir = "/home/work/users/PIL_ghj/LLM"


# =======================
# 허용된 알고리즘
# =======================
ALLOWED_ALGOS = {
    "Graph Algorithms",
    "Dynamic Programming",
    "Greedy Algorithms",
    "Sorting and Searching",
    "String Algorithms",
    "Data Structures",
    "Mathematics and Number Theory",
    "Simulation and Implementation",
}


# =======================
# pass@k 공식
# =======================
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


# =======================
# 파일 B: gold 알고리즘 집합
# =======================
def load_gold(benchmark_name):
    B_dir = f"datasets/{backend}_search_algorithm/{benchmark_dict[benchmark_name]}/{benchmark_name}_narrative_by_{backend}_search_algorithm.jsonl"
    qid_to_gold = {}
    with open(os.path.join(base_dir, B_dir), "r") as f:
        for line in f:
            obj = json.loads(line)
            golds = []
            for n in obj.get("narratives", []):
                algo = extract_algo_category(n)
                if algo and algo in ALLOWED_ALGOS:
                    golds.append(algo)
            if not golds:
                continue
            if gold_selection_mode == "unique":
                gold_set = set(golds)
                if len(gold_set) == 1:
                    qid_to_gold[obj["question_id"]] = list(gold_set)[0]
            elif gold_selection_mode == "majority":
                counter = Counter(golds)
                majority_algo, _ = counter.most_common(1)[0]
                qid_to_gold[obj["question_id"]] = majority_algo
    return qid_to_gold


# =======================
# 파일 A: original & narrative 로딩
# =======================
def load_results(benchmark_name, mode):
    A_dir = f"code/LiveCodeBench/algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}.jsonl"
    problems = {}
    with open(os.path.join(base_dir, A_dir)) as f:
        for line in f:
            obj = json.loads(line)
            problems[obj["question_id"]] = obj
    return problems


# =======================
# 분석 함수
# =======================
def analyze(benchmark_name):
    qid_to_gold = load_gold(benchmark_name)
    original_data = load_results(benchmark_name, "original")
    narrative_data = load_results(benchmark_name, "narrative")

    common_qids = set(original_data.keys()) & set(narrative_data.keys())

    results = {mode: {k: {"success": 0, "total": 0} for k in k_list} for mode in ["original", "narrative"]}
    dist = {mode: {k: Counter() for k in k_list} for mode in ["original", "narrative"]}

    skipped_qids = 0
    total_qids = 0

    for qid in common_qids:
        gold = qid_to_gold.get(qid, None)
        if not gold:
            skipped_qids += 1
            continue
        total_qids += 1

        data = {
            "original": original_data[qid]["results"],
            "narrative": narrative_data[qid]["results"],
        }

        if exclude_both_incorrect:
            orig_correct = any(s.get("is_correct", False) for s in data["original"])
            narr_correct = any(s.get("is_correct", False) for s in data["narrative"])
            if not orig_correct and not narr_correct:
                continue

        for mode in ["original", "narrative"]:
            if compare_mode == "with_io":
                samples = data[mode][:5]
            elif compare_mode == "merge":
                samples = data[mode][5:]
            else:
                samples = data[mode]

            n = len(samples)
            c = sum(s.get("is_correct", False) for s in samples)

            for k in k_list:
                results[mode][k]["total"] += 1

                if analysis_mode == "passk":
                    score = pass_at_k(n, c, k)
                    results[mode][k]["success"] += score

                    # breakdown: 기존 방식 (c > 0이면 match+correct)
                    if c > 0:
                        dist[mode][k]["match+correct"] += 1
                    else:
                        match_count = sum(extract_core_algorithm(s["analysis"]) == gold for s in samples)
                        match_ratio = match_count / n if n > 0 else 0
                        if match_ratio >= wrong_threshold:
                            dist[mode][k]["match+wrong"] += 1
                        else:
                            dist[mode][k]["mismatch+wrong"] += 1

                elif analysis_mode == "passk_soft":
                    score = pass_at_k(n, c, k)
                    results[mode][k]["success"] += score

                    match_count = sum(extract_core_algorithm(s["analysis"]) == gold for s in samples)
                    match_ratio = match_count / n if n > 0 else 0

                    if match_ratio >= soft_threshold and c > 0:
                        dist[mode][k]["match+correct"] += 1
                    elif c > 0:
                        dist[mode][k]["mismatch+correct"] += 1
                    else:
                        if match_ratio >= wrong_threshold:
                            dist[mode][k]["match+wrong"] += 1
                        else:
                            dist[mode][k]["mismatch+wrong"] += 1

                elif analysis_mode == "passk_prop":
                    score = pass_at_k(n, c, k)
                    results[mode][k]["success"] += score

                    match_count = sum(extract_core_algorithm(s["analysis"]) == gold for s in samples)
                    match_ratio = match_count / n if n > 0 else 0

                    # 비율로 기여
                    dist[mode][k]["match+correct"] += match_ratio
                    dist[mode][k]["match+wrong"] += (1 - match_ratio) * (c == 0)
                    # mismatch+wrong은 c==0 & match_ratio 낮을 때만
                    if c == 0 and match_ratio < wrong_threshold:
                        dist[mode][k]["mismatch+wrong"] += (1 - match_ratio)

    return results, dist, len(common_qids), total_qids, skipped_qids


# =======================
# 메인 실행
# =======================
if benchmark == "codeforces_both":
    sub_benchmarks = ["codeforces", "codeforces_challenging"]
else:
    sub_benchmarks = [benchmark]

agg_results = {mode: {k: {"success": 0, "total": 0} for k in k_list} for mode in ["original", "narrative"]}
agg_dist = {mode: {k: Counter() for k in k_list} for mode in ["original", "narrative"]}
agg_common, agg_total, agg_skipped = 0, 0, 0

for sub in sub_benchmarks:
    res, dist, common_qids, total_qids, skipped_qids = analyze(sub)

    for mode in ["original", "narrative"]:
        for k in k_list:
            agg_results[mode][k]["success"] += res[mode][k]["success"]
            agg_results[mode][k]["total"] += res[mode][k]["total"]
            agg_dist[mode][k].update(dist[mode][k])

    agg_common += common_qids
    agg_total += total_qids
    agg_skipped += skipped_qids


# =======================
# 출력
# =======================
for mode in ["original", "narrative"]:
    print(f"\n=== {analysis_mode} results ({mode}, compare={compare_mode}, benchmark={benchmark}) ===")
    for k in k_list:
        success = agg_results[mode][k]["success"]
        total = agg_results[mode][k]["total"]
        rate = success / total if total > 0 else 0
        print(f"{analysis_mode}@{k}: {success:.2f}/{total} = {rate:.2%}")
        print(f"--- Breakdown for {analysis_mode}@{k} ---")
        order = ["match+correct", "match+wrong", "mismatch+wrong", "mismatch+correct"]
        for case in order:
            v = agg_dist[mode][k][case]
            print(f"{case:>20}: {v} ({v/total:.2%})")

print(f"\n[Info] Common problems (sum of subs): {agg_common}")
print(f"[Info] Used problems after filtering: {agg_total}")
print(f"[Info] Skipped (no gold): {agg_skipped}")
print(f"[Info] exclude_both_incorrect={exclude_both_incorrect}")


import matplotlib.pyplot as plt
import numpy as np

# =======================
# (a), (b), (c) 매핑
# =======================
k = k_list[0]  # 현재는 k=10만 사용
def get_dist(mode):
    dist = agg_dist[mode][k]
    return {
        "a": dist["match+correct"],  # 옳은 알고리즘 선택
        "b": dist["match+wrong"],    # 세부 구현 오류
        "c": dist["mismatch+wrong"]  # 잘못된 알고리즘 선택
    }

dist_original = get_dist("original")
dist_narrative = get_dist("narrative")

# 전체 문제 수
total = agg_results["original"][k]["total"]

# 비율 변환
ratio_original = {k: v/total for k,v in dist_original.items()}
ratio_narrative = {k: v/total for k,v in dist_narrative.items()}

# =======================
# 시각화
# =======================
labels = ["(a) Correct Answer", 
          "(b) Implementation Error", 
          "(c) Wrong Algorithm"]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 7))

rects1 = ax.bar(x - width/2, [ratio_narrative[k] for k in ["a","b","c"]], 
                width, label="Narr.")
rects2 = ax.bar(x + width/2, [ratio_original[k] for k in ["a","b","c"]], 
                width, label="Orig.")

# 축/레이블
ax.set_ylabel("Ratio", fontsize=16)       # y축 라벨
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)   # x축 라벨들
ax.tick_params(axis='y', labelsize=14)    # y축 눈금 글씨
ax.tick_params(axis='x', labelsize=14)    # x축 눈금 글씨
ax.legend(fontsize=14)                    # 범례

# 값 표시
def autolabel(rects, fontsize=12):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=fontsize)    # 글씨 크기

autolabel(rects1, fontsize=14)
autolabel(rects2, fontsize=14)

plt.tight_layout()
plt.title("Contribution of Narrative Prompting to Algorithm Selection and Implementation", fontsize=16)
plt.show()



# %%






# %%
# --------------------------
# 특정 QID 입력
# --------------------------
TARGET_QID = "abc387_f"   # ✅ 여기 원하는 QID 넣으세요

# 파일 B: gold narratives 확인
gold_algos = []
with open(os.path.join(base_dir, B_dir), "r") as f:
    for line in f:
        obj = json.loads(line)
        if obj["question_id"] == TARGET_QID:
            for n in obj.get("narratives", []):
                algo = extract_algo_category(n)
                if algo:
                    gold_algos.append(algo)
            break

# 파일 A: 해당 QID의 결과 확인
with open(os.path.join(base_dir, A_dir)) as f:
    for line in f:
        obj = json.loads(line)
        if obj["question_id"] == TARGET_QID:
            with_io_samples = obj["results"][:5]
            merge_samples   = obj["results"][5:]

            with_io_preds = [extract_core_algorithm(s["analysis"]) for s in with_io_samples]
            merge_preds   = [extract_core_algorithm(s["analysis"]) for s in merge_samples]
            break

# --------------------------
# 표 출력
# --------------------------
print(f"\n[QID={TARGET_QID}] 비교 결과")

table = []
for i, algo in enumerate(gold_algos):
    table.append([f"B_narratives[{i}]", algo, ""])

for i, pred in enumerate(with_io_preds):
    table.append([f"A_with_io[{i}]", "", pred])

for i, pred in enumerate(merge_preds):
    table.append([f"A_merge[{i}]", "", pred])

print(tabulate(table, headers=["Source", "B (Gold Algorithm Category)", "A (Core Algorithm)"], tablefmt="grid"))
# %%
# %%
import os
import json
import re

base_dir = "/home/work/users/PIL_ghj/LLM"
A_dir = "code/LiveCodeBench/algorithm_analysis/algorithm_analysis_narrative_gemini.jsonl"
B_dir = "datasets/gemini_search_algorithm/LiveCodeBench/test6_narrative_by_gemini_search_algorithm.jsonl"

# ---------- 추출 함수 ----------
def extract_core_algorithm(text):
    if not text:
        return ""
    m = re.search(r"Core Algorithm:\s*(.+)", text)
    return m.group(1).strip() if m else ""

def extract_algo_category(text):
    m = re.search(r"Algorithm Category:\s*(.+)", text)
    return m.group(1).strip() if m else ""

# ---------- 파일 B: gold ----------
qid_to_B_algos = {}
with open(os.path.join(base_dir, B_dir), "r") as f:
    for line in f:
        obj = json.loads(line)
        qid = obj["question_id"]
        algos = []
        for n in obj.get("narratives", []):
            algo = extract_algo_category(n)
            if algo:
                algos.append(algo)
        qid_to_B_algos[qid] = algos

# ---------- 파일 A: pred ----------
qid_to_A_algos = {}
with open(os.path.join(base_dir, A_dir), "r") as f:
    for line in f:
        obj = json.loads(line)
        qid = obj["question_id"]
        algos = []
        for s in obj.get("results", []):
            algo = extract_core_algorithm(s["analysis"])
            if algo:
                algos.append(algo)
        qid_to_A_algos[qid] = algos

# ---------- JSON 저장 ----------
out_dir = os.path.join(base_dir, "code/LiveCodeBench/algorithm_analysis/algorithm_extraction_lists")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "A_extracted_algos.json"), "w", encoding="utf-8") as f:
    json.dump(qid_to_A_algos, f, indent=2, ensure_ascii=False)

with open(os.path.join(out_dir, "B_extracted_algos.json"), "w", encoding="utf-8") as f:
    json.dump(qid_to_B_algos, f, indent=2, ensure_ascii=False)

print(f"[Done] Saved A_extracted_algos.json and B_extracted_algos.json in {out_dir}")
# %%
