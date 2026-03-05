# %%
import os
import json
import pandas as pd
import math

# ===========================================================
# 경로 생성 함수
# ===========================================================
def get_eval_paths(mode, model, benchmark, directory_name, convert_option=None):
    paths = []

    # ===========================================================
    # ORIGINAL 경로 분기
    # ===========================================================
    if mode == "original":
        is_both = True if k == 10 else False
        if model == "Gemini-2.5-Flash":
            paths = [
                os.path.join(directory_name, "output", benchmark, "original_no_io", model,
                             "Scenario.codegeneration_5_0.2_eval.json"),
            ]
            if is_both:
                paths.append(
                    os.path.join(directory_name, "output", benchmark, 
                                 "original_no_io_0", model,
                                 "Scenario.codegeneration_5_0.2_eval.json")
                )

        else:
            raise ValueError(f"Unsupported model: {model}")

    # ===========================================================
    # NARRATIVE 경로 분기
    # ===========================================================
    elif mode == "narrative":
        if convert_option is None:
            raise ValueError("convert_option must be specified for narrative mode")

        # both = with_io + merge
        if convert_option == "both":
            convert_list = ["no_io", "merge_no_io"]
        else:
            convert_list = [convert_option]

        for convert in convert_list:
            for i in range(1, 6):
                paths.append(os.path.join(
                    directory_name, "output", benchmark,
                    f"narrative_{i}_{convert}",
                    model, "Scenario.codegeneration_1_0.2_eval.json"
                ))

    else:
        raise ValueError("mode must be either 'original' or 'narrative'")

    return paths


# ===========================================================
# 유틸 함수
# ===========================================================
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


# ===========================================================
# 기본 설정
# ===========================================================
base_dir = ""

models = [
    "Gemini-2.5-Flash",
]

benchmarks = [
    "codeforces_challenging",
    "codeforces",
    "humaneval_filtered",
    "test6",
]

convert_options = [
    # "both",
    "no_io", 
    "merge_no_io",
    
]

output_options = [
    "pass",
]

if "both" in convert_options:
    k = 10
elif "no_io" in convert_options or "merge_no_io" in convert_options:
    k = 5
else:
    raise

# ===========================================================
# 메인 루프
# ===========================================================
all_results = {}
for benchmark in benchmarks:
    all_results[benchmark] = {}

    for output_option in output_options:
        df = pd.DataFrame(index=models, columns=["original"] + convert_options)

        for model in models:
            # ======================
            # ORIGINAL
            # ======================
            orig_paths = get_eval_paths(
                "original", model, benchmark, base_dir
            )
            original_results = []
            for path in orig_paths:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        if "detail" in data[0]:
                            original_results.append(data[0]["detail"]["pass@1"])
                except Exception as e:
                    print(e)
                    pass

            if original_results:
                task_ids = original_results[0].keys()
                scores = []
                for tid in task_ids:
                    runs = [run.get(tid, 0) for run in original_results]
                    n = len(runs)
                    num_variant = k // n
                    c = round(sum(runs) * num_variant)
                    scores.append(pass_at_k(k, c, k))
                pass_score = sum(scores) / len(scores)
                df.loc[model, "original"] = round(pass_score * 100, 2)
            else:
                df.loc[model, "original"] = -1
                
            # ======================
            # NARRATIVE
            # ======================
            for convert_option in convert_options:
                narrative_paths = get_eval_paths(
                    "narrative", model, benchmark, base_dir, convert_option
                )

                narrative_results = []
                for path in narrative_paths:
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                            if "detail" in data[0]:
                                narrative_results.append(data[0]["detail"]["pass@1"])
                    except Exception:
                        pass

                if narrative_results:
                    task_ids = narrative_results[0].keys()
                    scores = []

                    for tid in task_ids:
                        runs = [run.get(tid, 0) for run in narrative_results]
                        n = len(runs)

                        num_variant = k // n
                        c = round(sum(runs) * num_variant)

                        scores.append(pass_at_k(k, c, k))

                    pass_score = sum(scores) / len(scores)
                    df.loc[model, convert_option] = round(pass_score * 100, 2)
                else:
                    df.loc[model, convert_option] = -1

        all_results[benchmark][output_option] = df


# ===========================================================
# 출력
# ===========================================================
for benchmark in benchmarks:
    if benchmark in ["codeforces_challenging", "codeforces"]:
        continue

    print(f"\n=== {benchmark} ===")
    for i, output_option in enumerate(output_options, 1):
        print(f"{i}. {output_option}")
        print(all_results[benchmark][output_option])

# ===========================================================
# codeforces weighted average (micro)
# ===========================================================
if "codeforces" in benchmarks and "codeforces_challenging" in benchmarks:
    combined_name = "codeforces (weighted)"
    print(f"\n=== {combined_name} ===")

    for i, output_option in enumerate(output_options, 1):
        df1 = all_results["codeforces_challenging"][output_option]
        df2 = all_results["codeforces"][output_option]

        # 벤치마크별 task 개수 계산
        # (각 결과 JSON에서 pass@1의 key 개수 = task 수)
        def count_tasks(model_name, benchmark_name, mode):
            paths = get_eval_paths(mode, model_name, benchmark_name, base_dir)
            total_tasks = 0
            for path in paths:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        if "detail" in data[0]:
                            total_tasks = max(total_tasks, len(data[0]["detail"]["pass@1"]))
                except Exception:
                    continue
            return total_tasks

        for model in models:
            n1 = count_tasks(model, "codeforces_challenging", "original")
            n2 = count_tasks(model, "codeforces", "original")
            n_total = n1 + n2

            if n_total == 0:
                print(f"[WARN] No tasks found for {model}")
                continue

            # 각 모델별 weighted average
            for col in ["original"] + convert_options:
                s1 = float(df1.loc[model, col])
                s2 = float(df2.loc[model, col])
                weighted_avg = (s1 * n1 + s2 * n2) / n_total
                df1.loc[model, col] = weighted_avg  # 재사용

        # 출력
        print(f"{i}. {output_option}")
        print(df1.astype(float).round(2))

# %%

# %% pass@5
# original
he_orig = (91.43-95.93)/95.93
lcb_orig = (37.71-50.16)/50.16
cf_orig = (31.32-52.24)/52.24

avg_orig = (he_orig + lcb_orig + cf_orig) / 3

he_narr = (96.19-96.19)/96.19
lcb_narr = (34.29-50.86)/50.86
cf_narr = (42.64-58.87)/58.87
avg_narr = (he_narr + lcb_narr + cf_narr) / 3

print(avg_orig*100)
print(avg_narr*100)
# %%
print(((91.43-95.93)+(37.71-50.16)+(31.32-52.24))/3)
print(((96.19-96.19)+(34.29-50.86)+(42.64-58.87))/3)

# %% pass@10
# original
he_orig = (93.33-96.19)/96.19
lcb_orig = (38.29-53.14)/53.14
cf_orig = (35.1-60.00)/60.00

avg_orig = (he_orig + lcb_orig + cf_orig) / 3

he_narr = (96.19-96.19)/96.19
lcb_narr = (52.00-57.14)/57.14
cf_narr = (57.36-67.55)/67.55
avg_narr = (he_narr + lcb_narr + cf_narr) / 3

print(avg_orig*100)
print(avg_narr*100)
# %%
print(((93.33-96.19)+(38.29-53.14)+(35.1-60.00))/3)
print(((96.19-96.19)+(52.00-57.14)+(57.36-67.55))/3)
# %%
import os
import json
import pandas as pd
import math


# ===========================================================
# 경로 생성 함수
# ===========================================================
def get_eval_paths(mode, model, benchmark, directory_name, convert_option=None, dataset="default", generator=None):
    paths = []

    # ===========================================================
    # codeforces_longer_v2 경로 분기
    # (오픈소스 narrative: output_search_algorithm_other_models_variant)
    # ===========================================================
    if dataset == "codeforces_longer_v2":
        if mode == "original":
            if model != "Gemini-2.5-Flash":
                paths = [os.path.join(
                    directory_name, "output", benchmark, "original", model,
                    "Scenario.codegeneration_10_0.2_eval.json"
                )]
            else:
                paths = [
                    os.path.join(directory_name, "output_search_algorithm",
                                 benchmark, "original", model,
                                 "Scenario.codegeneration_5_0.2_eval.json"),
                    os.path.join(directory_name, "output_search_algorithm",
                                 benchmark, "original_0", model,
                                 "Scenario.codegeneration_5_0.2_eval.json"),
                ]
            return paths

        elif mode == "narrative":
            if convert_option is None:
                raise ValueError("convert_option must be specified for narrative mode")

            convert_list = ["with_io", "merge"] if convert_option == "both" else [convert_option]

            if model == "Gemini-2.5-Flash":
                for convert in convert_list:
                    for i in range(1, 6):
                        paths.append(os.path.join(
                            directory_name,
                            "output_search_algorithm",
                            benchmark,
                            f"narrative_by_gemini_search_algorithm_narrative_{i}_{convert}",
                            model,
                            "Scenario.codegeneration_1_0.2_eval.json"
                        ))
            else:
                if generator is None:
                    raise ValueError("generator must be specified for codeforces_longer_v2 narrative mode with non-Gemini models")
                for convert in convert_list:
                    for i in range(1, 6):
                        paths.append(os.path.join(
                            directory_name,
                            "output_search_algorithm_other_models_variant",
                            benchmark,
                            f"{generator}_narrative_{i}_{convert}",
                            model,
                            "Scenario.codegeneration_1_0.2_eval.json"
                        ))

            return paths

        else:
            raise ValueError("mode must be either 'original' or 'narrative'")

    # ===========================================================
    # codeforces_longer 경로 분기
    # (오픈소스 narrative: output_search_algorithm, _eval_all.json)
    # ===========================================================
    elif dataset == "codeforces_longer":
        if mode == "original":
            if model != "Gemini-2.5-Flash":
                paths = [os.path.join(
                    directory_name, "output", benchmark, "original", model,
                    "Scenario.codegeneration_10_0.2_eval.json"
                )]
            else:
                paths = [
                    os.path.join(directory_name, "output_search_algorithm",
                                 benchmark, "original", model,
                                 "Scenario.codegeneration_5_0.2_eval.json"),
                    os.path.join(directory_name, "output_search_algorithm",
                                 benchmark, "original_0", model,
                                 "Scenario.codegeneration_5_0.2_eval.json"),
                ]
            return paths

        elif mode == "narrative":
            if convert_option is None:
                raise ValueError("convert_option must be specified for narrative mode")

            convert_list = ["with_io", "merge"] if convert_option == "both" else [convert_option]

            if model == "Gemini-2.5-Flash":
                filename = "Scenario.codegeneration_1_0.2_eval.json"
            else:
                filename = "Scenario.codegeneration_10_0.2_eval_all.json"

            for convert in convert_list:
                for i in range(1, 6):
                    paths.append(os.path.join(
                        directory_name,
                        "output_search_algorithm",
                        benchmark,
                        f"narrative_by_gemini_search_algorithm_narrative_{i}_{convert}",
                        model,
                        filename
                    ))

            return paths

        else:
            raise ValueError("mode must be either 'original' or 'narrative'")

    # ===========================================================
    # ORIGINAL 경로 분기 (default dataset)
    # ===========================================================
    if mode == "original":
        is_both = True if k == 10 else False
        if model == "Gemini-2.5-Flash":
            paths = [
                os.path.join(directory_name, "output", benchmark, "original_no_io", model,
                             "Scenario.codegeneration_5_0.2_eval.json"),
            ]
            if is_both:
                paths.append(
                    os.path.join(directory_name, "output", benchmark,
                                 "original_no_io_0", model,
                                 "Scenario.codegeneration_5_0.2_eval.json")
                )
        else:
            raise ValueError(f"Unsupported model: {model}")

    # ===========================================================
    # NARRATIVE 경로 분기 (default dataset)
    # ===========================================================
    elif mode == "narrative":
        if convert_option is None:
            raise ValueError("convert_option must be specified for narrative mode")

        if convert_option == "both":
            convert_list = ["no_io", "merge_no_io"]
        else:
            convert_list = [convert_option]

        for convert in convert_list:
            for i in range(1, 6):
                paths.append(os.path.join(
                    directory_name, "output", benchmark,
                    f"narrative_{i}_{convert}",
                    model, "Scenario.codegeneration_1_0.2_eval.json"
                ))

    else:
        raise ValueError("mode must be either 'original' or 'narrative'")

    return paths


# ===========================================================
# 유틸 함수
# ===========================================================
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def load_eval_results(path, dataset="default"):
    """
    경로에서 eval 결과를 로드하여 {task_id: score} dict로 반환.
    - detail.pass@1 포맷 (Gemini _eval.json, 오픈소스 _eval.json)
    - graded_list 포맷 (오픈소스 _eval_all.json) → graded_list[0] 사용
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Gemini 스타일 또는 오픈소스 _eval.json: detail.pass@1 dict
    if isinstance(data, list) and len(data) > 0 and "detail" in data[0]:
        return data[0]["detail"]["pass@1"]

    # 오픈소스 _eval_all.json: graded_list[0] 사용
    if isinstance(data, list) and len(data) > 0 and "graded_list" in data[0]:
        result = {}
        for i, item in enumerate(data):
            task_id = str(item.get("question_id", i))
            score = 1.0 if item["graded_list"][0] else 0.0
            result[task_id] = score
        print(result)
        return result

    return None


def get_k_for_convert_option(convert_option):
    """convert_option에 따라 k 결정 (original 제외)"""
    return 10 if convert_option == "both" else 5


def compute_scores_for_paths(paths, k_val, dataset="default", min_appearance=0):
    """
    min_appearance=0 : 필터링 없이 전체 task_id 사용 (기존 동작)
    min_appearance>0 : 해당 횟수 이상 등장한 task_id만 사용, pass_at_k(n, c, n)
    """
    from collections import Counter

    results = []
    for path in paths:
        try:
            res = load_eval_results(path, dataset=dataset)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"[WARN] {path}: {e}")

    if not results:
        return -1

    appearance = Counter()
    for res in results:
        for tid in res.keys():
            appearance[tid] += 1

    if min_appearance > 0:
        valid_task_ids = {tid for tid, cnt in appearance.items() if cnt >= min_appearance}
        if not valid_task_ids:
            print(f"[WARN] No task_ids with appearance >= {min_appearance}")
            return -1
        total = len(appearance)
        valid = len(valid_task_ids)
        if valid < total:
            print(f"[INFO] task_ids: {total} total → {valid} with appearance>={min_appearance} (excluded {total - valid})")
    else:
        valid_task_ids = set(appearance.keys())

    scores = []
    for tid in valid_task_ids:
        runs = [res[tid] for res in results if tid in res]
        n = len(runs)
        if min_appearance > 0:
            c = round(sum(runs))
            scores.append(pass_at_k(n, c, n))
        else:
            num_variant = k_val // n
            c = round(sum(runs) * num_variant)
            scores.append(pass_at_k(k_val, c, k_val))

    return round(sum(scores) / len(scores) * 100, 2)


# ===========================================================
# 기본 설정
# ===========================================================
base_dir = "/Users/jang-geonhui/Downloads/pil_llm_download/output_all_download"

models = [
    "Gemini-2.5-Flash",
]

benchmarks = [
    "codeforces_challenging",
    "codeforces",
    "humaneval_filtered",
    "test6",
]

# "original"을 포함하면 original 스코어도 계산
convert_options = [
    "original",
    "both",
    # "no_io",
    # "merge_no_io",
]

output_options = [
    "pass",
]

# codeforces_longer 전용 설정
CODEFORCES_LONGER_MODELS = [
    "Gemini-2.5-Flash",
    "DSCoder-6.7b-Ins",
    "DSCoder-V2-Lite-Instruct",
    "LLama3.1-8b-Ins",
    "Gemma-2-9b-Ins",
    "Gemma-2-27b-Ins",
    "Qwen2.5-Coder-Ins-7B",
    "Qwen2.5-Coder-Ins-32B",
    "Mistral-Small-24B-Instruct-2501",
]

CODEFORCES_LONGER_BENCHMARK = "codeforces_longer"
CODEFORCES_LONGER_CONVERT_OPTIONS = [
    "original",
    # "with_io",   # no_io → with_io
    # "merge",     # merge_no_io → merge
    "both",
]

# "codeforces_longer"  : 오픈소스 narrative → output_search_algorithm, _eval_all.json, graded_list[0]
# "codeforces_longer_v2": 오픈소스 narrative → output_search_algorithm_other_models_variant, _eval.json, detail.pass@1
CODEFORCES_LONGER_DATASET = "codeforces_longer_v2"

# codeforces_longer_v2일 때 narrative 생성 모델 선택 (오픈소스 모델에만 적용)
# CODEFORCES_LONGER_GENERATOR = "Mistral-Small-24B-Instruct-2501"
CODEFORCES_LONGER_GENERATOR = "Gemma-2-27b-Ins"
# CODEFORCES_LONGER_GENERATOR = "Qwen2.5-32B-Ins"
CODEFORCES_LONGER_MIN_APPEARANCE = 8  # 0 이면 교집합 없이 전체 사용 (기존 동작)

see_codeforces_longer_only = True

# k 결정 (default benchmark용)
narrative_options = [c for c in convert_options if c != "original"]
if "both" in narrative_options:
    k = 10
elif narrative_options:
    k = 5
else:
    k = 5  # original만 있는 경우 기본값


# ===========================================================
# 메인 루프 (기존 benchmarks)
# ===========================================================
if not see_codeforces_longer_only:
    all_results = {}
    for benchmark in benchmarks:
        all_results[benchmark] = {}

        for output_option in output_options:
            df = pd.DataFrame(index=models, columns=convert_options)

            for model in models:
                for convert_option in convert_options:
                    if convert_option == "original":
                        paths = get_eval_paths("original", model, benchmark, base_dir)
                        df.loc[model, "original"] = compute_scores_for_paths(paths, k)
                    else:
                        paths = get_eval_paths("narrative", model, benchmark, base_dir, convert_option)
                        df.loc[model, convert_option] = compute_scores_for_paths(paths, k)

            all_results[benchmark][output_option] = df

# ===========================================================
# codeforces_longer 루프
# ===========================================================
else:
    cl_results = {}
    for output_option in output_options:
        df = pd.DataFrame(
            index=CODEFORCES_LONGER_MODELS,
            columns=CODEFORCES_LONGER_CONVERT_OPTIONS
        )

        for model in CODEFORCES_LONGER_MODELS:
            for convert_option in CODEFORCES_LONGER_CONVERT_OPTIONS:
                if convert_option == "original":
                    paths = get_eval_paths(
                        "original", model, CODEFORCES_LONGER_BENCHMARK, base_dir,
                        dataset=CODEFORCES_LONGER_DATASET
                    )
                    df.loc[model, "original"] = compute_scores_for_paths(
                        paths, k_val=10, dataset=CODEFORCES_LONGER_DATASET
                    )
                else:
                    k_val = get_k_for_convert_option(convert_option)
                    narrative_paths = get_eval_paths(
                        "narrative", model, CODEFORCES_LONGER_BENCHMARK, base_dir,
                        convert_option, dataset=CODEFORCES_LONGER_DATASET,
                        generator=CODEFORCES_LONGER_GENERATOR
                    )
                    df.loc[model, convert_option] = compute_scores_for_paths(
                        narrative_paths, k_val, dataset=CODEFORCES_LONGER_DATASET,
                        min_appearance=CODEFORCES_LONGER_MIN_APPEARANCE  # ← 추가
                    )

        cl_results[output_option] = df


# ===========================================================
# 출력 (기존 benchmarks)
# ===========================================================
if not see_codeforces_longer_only:
    for benchmark in benchmarks:
        if benchmark in ["codeforces_challenging", "codeforces"]:
            continue

        print(f"\n=== {benchmark} ===")
        for i, output_option in enumerate(output_options, 1):
            print(f"{i}. {output_option}")
            print(all_results[benchmark][output_option])

    # ===========================================================
    # codeforces weighted average (micro)
    # ===========================================================
    if "codeforces" in benchmarks and "codeforces_challenging" in benchmarks:
        combined_name = "codeforces (weighted)"
        print(f"\n=== {combined_name} ===")

        for i, output_option in enumerate(output_options, 1):
            df1 = all_results["codeforces_challenging"][output_option]
            df2 = all_results["codeforces"][output_option]

            def count_tasks(model_name, benchmark_name, mode):
                paths = get_eval_paths(mode, model_name, benchmark_name, base_dir)
                total_tasks = 0
                for path in paths:
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                            if "detail" in data[0]:
                                total_tasks = max(total_tasks, len(data[0]["detail"]["pass@1"]))
                    except Exception:
                        continue
                return total_tasks

            for model in models:
                n1 = count_tasks(model, "codeforces_challenging", "original")
                n2 = count_tasks(model, "codeforces", "original")
                n_total = n1 + n2

                if n_total == 0:
                    print(f"[WARN] No tasks found for {model}")
                    continue

                for col in convert_options:
                    if float(df1.loc[model, col]) == -1 or float(df2.loc[model, col]) == -1:
                        continue
                    s1 = float(df1.loc[model, col])
                    s2 = float(df2.loc[model, col])
                    weighted_avg = (s1 * n1 + s2 * n2) / n_total
                    df1.loc[model, col] = weighted_avg

            print(f"{i}. {output_option}")
            print(df1.astype(float).round(2))

# ===========================================================
# codeforces_longer 출력
# ===========================================================
else:
    print(f"\n=== {CODEFORCES_LONGER_BENCHMARK} (dataset={CODEFORCES_LONGER_DATASET}"
          + (f", generator={CODEFORCES_LONGER_GENERATOR}" if CODEFORCES_LONGER_DATASET == "codeforces_longer_v2" else "")
          + ") ===")
    for i, output_option in enumerate(output_options, 1):
        print(f"{i}. {output_option}")
        df = cl_results[output_option].astype(float)

        # Gemini 제외한 오픈소스 모델 평균 행 추가
        opensource_models = [m for m in CODEFORCES_LONGER_MODELS if m != "Gemini-2.5-Flash"]
        avg_row = df.loc[opensource_models].mean().round(2)
        avg_row.name = "OpenSource Avg"

        print(pd.concat([df, avg_row.to_frame().T]).round(2))
# %%
