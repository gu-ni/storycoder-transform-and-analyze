# %%
import os
import json, re, math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

color_ours = "#4398B2"
color_bad = "#A09FAB"

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
compare_mode = "both"                     # "with_io", "merge", "both"
gold_source = "A_both"                         # ["B", "A_original", "A_narrative", "A_both"]
gold_selection_mode = "majority"          # "unique" or "majority"
analysis_mode = "passk_prop"              # "matching", "passk", "passk_soft", "passk_prop"
k_list = [10]                             # k 값들


# 새로운 플래그
correct_algo_match_threshold = 0.5        # match+correct 판정 기준 (passk_soft 모드에서 사용)
wrong_algo_match_threshold = 0.5          # match+wrong vs mismatch+wrong 기준
correct_success_threshold = 0.0

exclude_both_incorrect = True             # original & narrative 둘 다 틀리면 제외
exclude_both_correct = True               # original & narrative 둘 다 모든 샘플 정답이면 제외


simple_when_correct = True
simple_when_wrong = True

print_table = False
plot_with_sub_total = True
plot_list = ["a","b","c"]

# =======================
# 메인 실행
# =======================

benchmarks = [
    # "humaneval_filtered", 
    "test6", 
    "codeforces_both"
]

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

base_dir = "/Users/jang-geonhui/Downloads/pil_llm_download"


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
# gold 알고리즘 집합
# =======================
def load_gold(benchmark_name, original_data=None, narrative_data=None):
    qid_to_gold = {}
    total_candidates = 0
    used_candidates = 0

    if gold_source == "B":
        # 기존 방식
        B_dir = f"{backend}_search_algorithm/{benchmark_dict[benchmark_name]}/{benchmark_name}_narrative_by_{backend}_search_algorithm.jsonl"
        with open(os.path.join(base_dir, B_dir), "r") as f:
            for line in f:
                obj = json.loads(line)
                golds = []
                for n in obj.get("narratives", []):
                    algo = extract_algo_category(n)
                    if algo:
                        golds.append(algo)
                if not golds:
                    continue
                total_candidates += 1
                if gold_selection_mode == "unique":
                    gold_set = set(golds)
                    if len(gold_set) == 1:
                        qid_to_gold[obj["question_id"]] = list(gold_set)[0]
                        used_candidates += 1
                elif gold_selection_mode == "majority":
                    counter = Counter(golds)
                    majority_algo, _ = counter.most_common(1)[0]
                    qid_to_gold[obj["question_id"]] = majority_algo
                    used_candidates += 1

    else:
        # 파일 A 기반
        modes_to_check = []
        if gold_source == "A_original":
            modes_to_check = [("original", original_data)]
        elif gold_source == "A_narrative":
            modes_to_check = [("narrative", narrative_data)]
        elif gold_source == "A_both":
            modes_to_check = [("original", original_data), ("narrative", narrative_data)]

        for mode, data in modes_to_check:
            for qid, obj in data.items():
                correct_algos = []
                for s in obj["results"]:
                    if s.get("is_correct", False):
                        algo = extract_core_algorithm(s.get("analysis", ""))
                        # if algo:
                        if algo and algo != "N/A":
                            correct_algos.append(algo)

                if not correct_algos:
                    continue
                total_candidates += 1

                if gold_selection_mode == "unique":
                    gold_set = set(correct_algos)
                    if len(gold_set) == 1:
                        qid_to_gold[qid] = list(gold_set)[0]
                        used_candidates += 1
                elif gold_selection_mode == "majority":
                    counter = Counter(correct_algos)
                    majority_algo, _ = counter.most_common(1)[0]
                    qid_to_gold[qid] = majority_algo
                    used_candidates += 1

    gold_stats = {
        "total_candidates": total_candidates,
        "used_candidates": used_candidates,
        "skipped_candidates": total_candidates - used_candidates
    }
    return qid_to_gold, gold_stats


# =======================
# 파일 A: original & narrative 로딩
# =======================
def load_results(benchmark_name, mode):
    # A_dir = f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}_edited.jsonl"
    # A_dir = f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}.jsonl"
    A_dir = f"algorithm_analysis_fixed/algorithm_analysis_{benchmark_name}_{mode}_{backend}.jsonl"

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
    original_data = load_results(benchmark_name, "original")
    narrative_data = load_results(benchmark_name, "narrative")

    # gold을 파일 B 또는 A에서 추출
    qid_to_gold, gold_stats = load_gold(benchmark_name, original_data, narrative_data)

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
        
        if exclude_both_correct:
            all_orig_correct = all(s.get("is_correct", False) for s in data["original"])
            all_narr_correct = all(s.get("is_correct", False) for s in data["narrative"])
            if all_orig_correct and all_narr_correct:
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
                        if match_ratio >= wrong_algo_match_threshold:
                            dist[mode][k]["match+wrong"] += 1
                        else:
                            dist[mode][k]["mismatch+wrong"] += 1

                elif analysis_mode == "passk_soft":
                    score = pass_at_k(n, c, k)
                    results[mode][k]["success"] += score

                    match_count = sum(extract_core_algorithm(s["analysis"]) == gold for s in samples)
                    match_ratio = match_count / n if n > 0 else 0
                    
                    cond_correct = (c / n) > correct_success_threshold
                    if cond_correct:
                        if match_ratio >= correct_algo_match_threshold:
                            dist[mode][k]["match+correct"] += 1
                        else:
                            dist[mode][k]["mismatch+correct"] += 1
                    
                    else:
                        if match_ratio >= wrong_algo_match_threshold:
                            dist[mode][k]["match+wrong"] += 1
                        else:
                            dist[mode][k]["mismatch+wrong"] += 1

                elif analysis_mode == "passk_prop":
                    score = pass_at_k(n, c, k)
                    results[mode][k]["success"] += score

                    match_count = sum(extract_core_algorithm(s["analysis"]) == gold for s in samples)
                    match_ratio = match_count / n if n > 0 else 0
                    
                    cond_correct = (c / n) > correct_success_threshold
                    if cond_correct:
                        if simple_when_correct:
                            # 정답 있음 → correct 계열에 비율로 분배
                            dist[mode][k]["match+correct"]    += match_ratio
                            dist[mode][k]["mismatch+correct"] += (1 - match_ratio)
                        else:
                            if match_ratio >= correct_algo_match_threshold:
                                dist[mode][k]["match+correct"] += match_ratio
                            else:
                                dist[mode][k]["mismatch+correct"] += (1 - match_ratio)
                    
                    else:
                        if simple_when_wrong:
                            dist[mode][k]["match+wrong"] += match_ratio
                            dist[mode][k]["mismatch+wrong"] += (1 - match_ratio)
                        else:
                            # 정답 없음 → wrong 계열에서 threshold로 분리
                            if match_ratio >= wrong_algo_match_threshold:
                                dist[mode][k]["match+wrong"] += match_ratio
                            else:
                                dist[mode][k]["mismatch+wrong"] += (1 - match_ratio)

    return results, dist, len(common_qids), total_qids, skipped_qids, gold_stats


for idx, benchmark in enumerate(benchmarks):
    if benchmark == "codeforces_both":
        sub_benchmarks = ["codeforces", "codeforces_challenging"]
    else:
        sub_benchmarks = [benchmark]

    agg_results = {mode: {k: {"success": 0, "total": 0} for k in k_list} for mode in ["original", "narrative"]}
    agg_dist = {mode: {k: Counter() for k in k_list} for mode in ["original", "narrative"]}
    agg_common, agg_total, agg_skipped = 0, 0, 0

    for sub in sub_benchmarks:
        res, dist, common_qids, total_qids, skipped_qids, gold_stats = analyze(sub)
        
        if print_table:
            print(f"\n[Gold Info] gold_source={gold_source}, gold_selection_mode={gold_selection_mode}")
            print(f"[Gold Info] Total candidates checked: {gold_stats['total_candidates']}")
            print(f"[Gold Info] Used (valid gold found): {gold_stats['used_candidates']}")
            print(f"[Gold Info] Skipped (invalid or empty): {gold_stats['skipped_candidates']}")


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
        if print_table:
            print(f"\n=== {analysis_mode} results ({mode}, compare={compare_mode}, benchmark={benchmark}) ===")
        for k in k_list:
            success = agg_results[mode][k]["success"]
            total = agg_results[mode][k]["total"]
            rate = success / total if total > 0 else 0
            if print_table:
                print(f"{analysis_mode}@{k}: {success:.2f}/{total} = {rate:.2%}")
                print(f"--- Breakdown for {analysis_mode}@{k} ---")
            order = ["match+correct", "match+wrong", "mismatch+wrong", "mismatch+correct"]
            for case in order:
                v = agg_dist[mode][k][case]
                if print_table:
                    print(f"{case:>20}: {round(v, 3)} ({v/total:.2%})")
    
    if print_table:
        print(f"\n[Info] Common problems (sum of subs): {agg_common}")
        print(f"[Info] Used problems after filtering: {agg_total}")
        print(f"[Info] Skipped (no gold): {agg_skipped}")
        print(f"[Info] exclude_both_incorrect={exclude_both_incorrect}")

    # =======================
    # (a), (b), (c) 매핑
    # =======================
    k = k_list[0]  # 현재는 k=10만 사용
    def get_dist(mode):
        dist = agg_dist[mode][k]
        return {
            "a": dist["match+correct"],  # 옳은 알고리즘 선택
            "b": dist["match+wrong"],    # 세부 구현 오류
            "c": dist["mismatch+wrong"], # 잘못된 알고리즘 선택
            "d": dist["mismatch+correct"]
        }

    dist_original = get_dist("original")
    dist_narrative = get_dist("narrative")

    # 전체 문제 수
    total = agg_results["original"][k]["total"]

    # 비율 변환
    if plot_with_sub_total:
        # plot_list에 포함된 subset만 합산해서 normalization
        sum_original = sum(dist_original[p] for p in plot_list)
        sum_narrative = sum(dist_narrative[p] for p in plot_list)

        ratio_original = {p: dist_original[p] / sum_original if sum_original > 0 else 0 for p in plot_list}
        ratio_narrative = {p: dist_narrative[p] / sum_narrative if sum_narrative > 0 else 0 for p in plot_list}
    else:
        ratio_original = {p: dist_original[p] / total if total > 0 else 0 for p in plot_list}
        ratio_narrative = {p: dist_narrative[p] / total if total > 0 else 0 for p in plot_list}

    # =======================
    # 시각화
    # =======================
    labels = ["(i) Correct\nSolution", 
            "(ii) Implement-\nation Error", 
            "(iii) Wrong\nAlgorithm",
            "(vi) False\nCorrect"]

    x = np.arange(len(plot_list))
    width = 0.35
    
    if idx == 0:
        fig, ax = plt.subplots(figsize=(9, 8))
        font_ratio = 2.2
    else:
        fig, ax = plt.subplots(figsize=(8.5, 8))
        font_ratio = 2.2
    
    rects1 = ax.bar(x - width/2, [ratio_original[k] for k in plot_list], 
                width, label="Repeated Sampling", color=color_bad, 
                hatch='/', edgecolor='white')
    rects2 = ax.bar(x + width/2, [ratio_narrative[k] for k in plot_list], 
                width, label="Narrative (Ours)", color=color_ours)


    # 축/레이블
    if idx == 0:
        ax.set_ylabel("Ratio", fontsize=16*font_ratio)       # y축 라벨
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(plot_list)], fontsize=6*font_ratio)   # x축 라벨들
    ax.tick_params(axis='y', labelsize=14*font_ratio)    # y축 눈금 글씨
    ax.tick_params(axis='x', labelsize=14*font_ratio)    # x축 눈금 글씨
    if idx == 0:
        ax.legend(fontsize=14*font_ratio)                    # 범례

    # 값 표시
    def autolabel(rects, fontsize):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=fontsize)  # 글씨 크기

    autolabel(rects1, fontsize=14*font_ratio)
    autolabel(rects2, fontsize=14*font_ratio)
    
    ymax = max([r.get_height() for r in rects1] + [r.get_height() for r in rects2])
    ax.set_ylim(0, ymax * 1.1)
    
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=1*font_ratio)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.1*font_ratio)
    plt.tight_layout()
    plt.title(benchmark_dict[benchmark], fontsize=16*font_ratio)
    plt.show()


# %%

# %%





# %%
import os, json, re
from collections import Counter

base_dir = "/Users/jang-geonhui/Downloads/pil_llm_download"

def extract_core_algorithm(text):
    if not text:
        return ""
    m = re.search(r"Core Algorithm:\s*(.+)", text)
    return m.group(1).strip() if m else ""

def extract_algo_category(text):
    m = re.search(r"Algorithm Category:\s*(.+)", text)
    return m.group(1).strip() if m else ""

def load_gold_from_B(benchmark_name, backend="gemini", gold_selection_mode="majority"):
    """파일 B에서 gold 알고리즘 추출"""
    qid_to_gold = {}
    B_path = f"{backend}_search_algorithm/{benchmark_dict[benchmark_name]}/{benchmark_name}_narrative_by_{backend}_search_algorithm.jsonl"
    with open(os.path.join(base_dir, B_path)) as f:
        for line in f:
            obj = json.loads(line)
            algos = [extract_algo_category(n) for n in obj.get("narratives", []) if extract_algo_category(n)]
            if not algos:
                continue
            if gold_selection_mode == "majority":
                counter = Counter(algos)
                majority_algo, _ = counter.most_common(1)[0]
                qid_to_gold[obj["question_id"]] = majority_algo
            elif gold_selection_mode == "unique":
                if len(set(algos)) == 1:
                    qid_to_gold[obj["question_id"]] = algos[0]
    return qid_to_gold

def load_A_results(benchmark_name, mode, backend="gemini"):
    """파일 A에서 original/narrative 결과 불러오기"""
    # A_path = f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}_edited.jsonl"
    A_path = f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}.jsonl"

    results = {}
    with open(os.path.join(base_dir, A_path)) as f:
        for line in f:
            obj = json.loads(line)
            results[obj["question_id"]] = obj["results"]
    return results

def compute_match_ratio(
    benchmark_name, mode, backend="gemini", 
    gold_source="B", gold_selection_mode="majority",
    metric_mode="response"
):
    """
    mode: "original" or "narrative"
    gold_source:
      - "B" : 파일 B에서 추출한 gold과 비교
      - "A" : 실제 정답 응답들의 다수결/unique 알고리즘을 gold으로 삼음
    gold_selection_mode:
      - "majority" : 가장 많이 등장한 알고리즘
      - "unique"   : 모든 정답 응답의 알고리즘이 동일할 때만 gold 인정
    metric_mode:
      - "response" : 정답 응답 개수 기준 (기존 방식)
      - "problem"  : 문제 단위 pass@k 방식
    """
    data = load_A_results(benchmark_name, mode, backend)

    # gold 선택
    if gold_source == "B":
        qid_to_gold = load_gold_from_B(benchmark_name, backend, gold_selection_mode)
    elif gold_source == "A":
        qid_to_gold = {}
        for qid, samples in data.items():
            correct_algos = [extract_core_algorithm(s.get("analysis", "")) 
                             for s in samples if s.get("is_correct", False) and extract_core_algorithm(s.get("analysis", ""))]
            if not correct_algos:
                continue
            if gold_selection_mode == "majority":
                counter = Counter(correct_algos)
                majority_algo, _ = counter.most_common(1)[0]
                qid_to_gold[qid] = majority_algo
            elif gold_selection_mode == "unique":
                if len(set(correct_algos)) == 1:
                    qid_to_gold[qid] = correct_algos[0]
    else:
        raise ValueError("gold_source must be 'B' or 'A'")

    # ============================
    # metric_mode에 따른 계산 분기
    # ============================
    if metric_mode == "response":
        total_correct = 0
        match_correct = 0
        for qid, samples in data.items():
            gold = qid_to_gold.get(qid, None)
            if not gold:
                continue
            for s in samples:
                if s.get("code", "") == "":  # 빈 코드 샘플 스킵
                    continue
                if s.get("is_correct", False):
                    algo = extract_core_algorithm(s.get("analysis", ""))
                    # if algo:
                    if algo and algo != "N/A":
                        total_correct += 1
                        if algo == gold:
                            match_correct += 1
        ratio = match_correct / total_correct if total_correct > 0 else 0
        return {
            "benchmark": benchmark_name,
            "mode": mode,
            "gold_source": gold_source,
            "gold_selection_mode": gold_selection_mode,
            "metric_mode": metric_mode,
            "total_correct": total_correct,
            "match_correct": match_correct,
            "ratio": ratio
        }

    elif metric_mode == "problem":
        total_problems_with_correct = 0
        match_correct_problems = 0
        for qid, samples in data.items():
            gold = qid_to_gold.get(qid, None)
            if not gold:
                continue
            correct_algos = [extract_core_algorithm(s.get("analysis", "")) 
                             for s in samples if s.get("is_correct", False) and extract_core_algorithm(s.get("analysis", ""))]
            if not correct_algos:
                continue
            total_problems_with_correct += 1
            if any(algo == gold for algo in correct_algos):
                match_correct_problems += 1
        ratio = match_correct_problems / total_problems_with_correct if total_problems_with_correct > 0 else 0
        return {
            "benchmark": benchmark_name,
            "mode": mode,
            "gold_source": gold_source,
            "gold_selection_mode": gold_selection_mode,
            "metric_mode": metric_mode,
            "total_problems_with_correct": total_problems_with_correct,
            "match_correct_problems": match_correct_problems,
            "ratio": ratio
        }
    else:
        raise ValueError("metric_mode must be 'response' or 'problem'")

# ============================
# 사용 예시
# ============================
benchmarks = ["humaneval_filtered", "test6", "codeforces", "codeforces_challenging"]

gold_source = "B"       # ["A", "B", "both"]
sel_mode = "majority"     # ["majority", "unique"]
metric_mode = "problem" # ["response", "problem"]

print(f"sel_mode: {sel_mode}, metric_mode: {metric_mode}")
for bm in benchmarks:
    for mode in ["original", "narrative"]:
        line = f"[{bm}][{mode}]"
        if gold_source in ["A", "both"]:
            stats_A = compute_match_ratio(
                bm, mode, backend="gemini",
                gold_source="A",
                gold_selection_mode=sel_mode,
                metric_mode=metric_mode
            )
            line += f" A기준 {stats_A['match_correct' if metric_mode=='response' else 'match_correct_problems']} / {stats_A['total_correct' if metric_mode=='response' else 'total_problems_with_correct']} ({stats_A['ratio']:.2%})"
        if gold_source in ["B", "both"]:
            stats_B = compute_match_ratio(
                bm, mode, backend="gemini",
                gold_source="B",
                gold_selection_mode=sel_mode,
                metric_mode=metric_mode
            )
            line += f" | B기준 {stats_B['match_correct' if metric_mode=='response' else 'match_correct_problems']} / {stats_B['total_correct' if metric_mode=='response' else 'total_problems_with_correct']} ({stats_B['ratio']:.2%})"
        print(line)


# original 모드 파일에서 question_id 개수 세기
original_data = load_A_results("test6", "original", backend="gemini")
total_problems_test6 = len(original_data)

original_data = load_A_results("codeforces", "original", backend="gemini")
total_problems_codeforces = len(original_data)

original_data = load_A_results("codeforces_challenging", "original", backend="gemini")
total_problems_codeforces_challenging = len(original_data)

total_problems = {
    "humaneval_filtered": len(load_A_results("humaneval_filtered", "original")),
    "test6": len(load_A_results("test6", "original")),
    "codeforces": len(load_A_results("codeforces", "original")),
    "codeforces_challenging": len(load_A_results("codeforces_challenging", "original"))
}

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams["font.family"] = "Times New Roman"

# 결과 수집
results = {}
for bm in benchmarks:
    results[bm] = {}
    for mode in ["narrative", "original"]:
        stats_B = compute_match_ratio(
            bm, mode, backend="gemini",
            gold_source="B",
            gold_selection_mode=sel_mode,
            metric_mode=metric_mode
        )
        results[bm][mode] = stats_B

# codeforces + codeforces_challenging 합치기
def average_benchmarks(b1, b2, new_name="codeforces_avg"):
    merged = {}
    total_all = total_problems[b1] + total_problems[b2]
    for mode in ["original", "narrative"]:
        t1, m1 = results[b1][mode]["total_problems_with_correct"], results[b1][mode]["match_correct_problems"]
        t2, m2 = results[b2][mode]["total_problems_with_correct"], results[b2][mode]["match_correct_problems"]

        total = t1 + t2
        match = m1 + m2
        ratio = match / total if total > 0 else 0

        merged[mode] = {
            "total_problems_with_correct": total,
            "match_correct_problems": match,
            "ratio": ratio,
            "coverage": total / total_all * 100   # coverage % 추가
        }
    results[new_name] = merged
    total_problems[new_name] = total_all

# 평균 benchmark 추가
average_benchmarks("codeforces", "codeforces_challenging", new_name="codeforces_avg")

# Plot
fig, ax = plt.subplots(figsize=(7,6))
plot_benchmarks = ["humaneval_filtered", "test6", "codeforces_avg"]
font_ratio = 1

for bm in plot_benchmarks:
    orig = results[bm]["original"]
    narr = results[bm]["narrative"]

    # coverage %로 변환
    x_orig, y_orig = orig["total_problems_with_correct"] / total_problems[bm] * 100, orig["ratio"]*100
    x_narr, y_narr = narr["total_problems_with_correct"] / total_problems[bm] * 100, narr["ratio"]*100
    
    print(f"benchmark: {bm}\nx_orig: {x_orig}, x_narr: {x_narr}\ny_orig: {y_orig}, y_narr: {y_narr}")
    
    print((y_narr - y_orig), "\n")

    # 화살표 (annotate → 찌그러짐 방지, 머리 크기 mutation_scale로 조절)
    ax.annotate("",
        xy=(x_narr, y_narr), xycoords="data",
        xytext=(x_orig, y_orig), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#BFBFBF", lw=3,
                        alpha=1, mutation_scale=40)
    )
    
     # 점 표시
    ax.scatter(x_narr, y_narr, color=color_ours, marker="o", s=380*font_ratio,
               label="Narrative (Ours)" if bm==plot_benchmarks[0] else "")
    ax.scatter(x_orig, y_orig, color=color_bad, marker="s", s=340*font_ratio,
               label="Repeated Sampling" if bm==plot_benchmarks[0] else "")

    # 라벨
    if bm == "test6":
        bm_name = "LiveCodeBench"
        x_d, y_d = 2.5, 0
    elif bm == "codeforces_avg":
        bm_name = "CodeForces"
        x_d, y_d = -4.9, 0.7
    elif bm == "humaneval_filtered":
        bm_name = "HumanEval"
        x_d, y_d = -16.5, 0.8
    ax.text(x_narr + x_d, y_narr + y_d, bm_name, fontsize=30*font_ratio)

# tight view (데이터 주변 확대)
x_vals, y_vals = [], []
for bm in plot_benchmarks:
    for mode in ["original", "narrative"]:
        x_vals.append(results[bm][mode]["total_problems_with_correct"] / total_problems[bm] * 100)
        y_vals.append(results[bm][mode]["ratio"]*100)

x_min, x_max = min(x_vals), max(x_vals)
y_min, y_max = min(y_vals), max(y_vals)
x_margin, y_margin = (x_max - x_min)*0.1, (y_max - y_min)*0.1
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)

ax.xaxis.set_major_locator(MultipleLocator(10))

for spine in ax.spines.values():
    spine.set_linewidth(2*font_ratio)

ax.set_xlabel("Pass@10 (%)", fontsize=27*font_ratio)
ax.set_ylabel("Agreement Ratio (%)", fontsize=27*font_ratio)
ax.tick_params(axis='x', labelsize=26*font_ratio)
ax.tick_params(axis='y', labelsize=26*font_ratio)
# ax.set_title("Narrative Improves Coverage and Alignment", fontsize=20)
ax.legend(fontsize=20*font_ratio)
ax.grid(True, linestyle="--", alpha=0.3, linewidth=2*font_ratio)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# %%
import os, json, re, math
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# =======================
# 공통 함수
# =======================
def extract_core_algorithm(text):
    if not text:
        return ""
    m = re.search(r"Core Algorithm:\s*(.+)", text)
    return m.group(1).strip() if m else ""

def extract_algo_category(text):
    m = re.search(r"Algorithm Category:\s*(.+)", text)
    return m.group(1).strip() if m else ""

benchmark_dict = {
    "humaneval_filtered": "HumanEval",
    "test6": "LiveCodeBench",
    "codeforces": "CodeForces",
    "codeforces_challenging": "CodeForces",
    "codeforces_both": "CodeForces",
}

base_dir = "/Users/jang-geonhui/Downloads/pil_llm_download"

# =======================
# 옵션 설정
# =======================
gold_selection_mode = "majority"          # "unique" or "majority"
analysis_mode = "passk_prop"              # "matching", "passk", "passk_soft", "passk_prop"
k_list = [10]                             # k 값들

sel_mode = "majority"     # ["majority", "unique"]  (for 3)
metric_mode = "problem"   # ["response", "problem"] (for 3)

# =======================
# 파일 로딩
# =======================
def load_results(benchmark_name, mode):
    """파일 A: original/narrative"""
    A_dir = f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_gemini_edited.jsonl"
    problems = {}
    with open(os.path.join(base_dir, A_dir)) as f:
        for line in f:
            obj = json.loads(line)
            problems[obj["question_id"]] = obj
    return problems

def load_gold_from_B(benchmark_name, gold_selection_mode="majority"):
    """파일 B: original 문제 기반 예상 알고리즘"""
    qid_to_gold = {}
    B_path = f"gemini_search_algorithm/{benchmark_dict[benchmark_name]}/{benchmark_name}_narrative_by_gemini_search_algorithm.jsonl"
    with open(os.path.join(base_dir, B_path)) as f:
        for line in f:
            obj = json.loads(line)
            algos = [extract_algo_category(n) for n in obj.get("narratives", []) if extract_algo_category(n)]
            if not algos:
                continue
            if gold_selection_mode == "majority":
                counter = Counter(algos)
                majority_algo, _ = counter.most_common(1)[0]
                qid_to_gold[obj["question_id"]] = majority_algo
            elif gold_selection_mode == "unique":
                if len(set(algos)) == 1:
                    qid_to_gold[obj["question_id"]] = algos[0]
    return qid_to_gold

# =======================
# 2. 정답 코드 기반 알고리즘 추출
# =======================
def load_gold_from_A(benchmark_name, gold_source="A_original"):
    qid_to_gold = {}
    if gold_source == "A_original":
        data = load_results(benchmark_name, "original")
    elif gold_source == "A_narrative":
        data = load_results(benchmark_name, "narrative")
    elif gold_source == "A_both":
        data = {}
        data.update(load_results(benchmark_name, "original"))
        data.update(load_results(benchmark_name, "narrative"))
    else:
        raise ValueError("gold_source must be A_original, A_narrative, or A_both")

    for qid, obj in data.items():
        correct_algos = []
        for s in obj["results"]:
            if s.get("is_correct", False):
                algo = extract_core_algorithm(s.get("analysis", ""))
                # if algo:
                if algo and algo != "N/A":
                    correct_algos.append(algo)
        if not correct_algos:
            continue
        if gold_selection_mode == "unique":
            if len(set(correct_algos)) == 1:
                qid_to_gold[qid] = correct_algos[0]
        elif gold_selection_mode == "majority":
            counter = Counter(correct_algos)
            majority_algo, _ = counter.most_common(1)[0]
            qid_to_gold[qid] = majority_algo
    return qid_to_gold

# =======================
# 2 vs 3 비교
# =======================
def compare_algorithms(benchmark_name, gold_source="A_original"):
    # gold from 2 (정답 코드 기반)
    gold_A = load_gold_from_A(benchmark_name, gold_source)

    # gold from 3 (original 문제 기반 예상 알고리즘)
    gold_B = load_gold_from_B(benchmark_name, sel_mode)

    total = 0
    match = 0

    for qid in gold_A.keys():
        if qid in gold_B:
            total += 1
            if gold_A[qid] == gold_B[qid]:
                match += 1

    ratio = match / total if total > 0 else 0
    return {
        "benchmark": benchmark_name,
        "gold_source": gold_source,
        "total": total,
        "match": match,
        "ratio": ratio
    }

# =======================
# 실행
# =======================
benchmarks = ["test6", "codeforces", "codeforces_challenging"]

for bm in benchmarks:
    for gs in ["A_original", "A_narrative", "A_both"]:
        stats = compare_algorithms(bm, gold_source=gs)
        print(f"[{bm}][{gs}] {stats['match']} / {stats['total']} ({stats['ratio']:.2%})")

# =======================
# 간단 플롯 (bar chart)
# =======================
def plot_results(benchmarks, sources):
    fig, ax = plt.subplots(figsize=(8,6))
    x = np.arange(len(benchmarks))
    width = 0.25

    for i, gs in enumerate(sources):
        vals = []
        for bm in benchmarks:
            stats = compare_algorithms(bm, gold_source=gs)
            vals.append(stats["ratio"]*100)
        ax.bar(x + i*width, vals, width, label=gs)

    ax.set_xticks(x + width)
    ax.set_xticklabels([benchmark_dict[bm] for bm in benchmarks], fontsize=14)
    ax.set_ylabel("Agreement (2 vs 3) [%]", fontsize=14)
    ax.set_title("Comparison of Correct-Solution Algorithms (2) vs. Predicted Algorithms (3)", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_results(["test6", "codeforces", "codeforces_challenging"], ["A_original", "A_narrative", "A_both"])



# %%
