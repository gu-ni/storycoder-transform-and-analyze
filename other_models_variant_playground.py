# %%
import re
from collections import defaultdict
import pandas as pd

log_file = "log.txt"

pattern_model = re.compile(r"Processing model: (.+?) ====================")
pattern_found = re.compile(r"Found (\d+) problems.*")
pattern_saved = re.compile(
    r"Saved .*/([^/]+)/[^/]+\.jsonl \(removed (\d+) too_long, (\d+) too_short, (\d+) missing_section\)"
)

stats = defaultdict(lambda: defaultdict(lambda: {
    "total": 0, "too_long": 0, "too_short": 0, "missing": 0, "original": 0
}))

current_model = None
current_benchmark = None

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()

        # 모델명
        m = pattern_model.search(line)
        if m:
            current_model = m.group(1)
            continue

        # Found X problems
        fnd = pattern_found.search(line)
        if fnd and current_model:
            num_problems = int(fnd.group(1))
            current_benchmark = None  # 초기화
            continue

        # Saved ... (removed ...)
        s = pattern_saved.search(line)
        if s and current_model:
            benchmark, tl, ts, ms = s.groups()
            tl, ts, ms = map(int, (tl, ts, ms))

            # 한 번만 original 개수 등록
            if stats[current_model][benchmark]["original"] == 0:
                # 한 벤치마크 당 문제 수는 Found에서 잡히는데,
                # narratives는 항상 5개씩 생성된다고 했으므로 ×5
                stats[current_model][benchmark]["original"] = num_problems * 5

            stats[current_model][benchmark]["too_long"] += tl
            stats[current_model][benchmark]["too_short"] += ts
            stats[current_model][benchmark]["missing"] += ms
            stats[current_model][benchmark]["total"] += (tl + ts + ms)

# DataFrame 변환
rows = []
for model, benchmarks in stats.items():
    for benchmark, values in benchmarks.items():
        total = values["total"]
        original = values["original"]
        rows.append({
            "Model": model,
            "Benchmark": benchmark,
            "Original Count": original,
            "Too Long": values["too_long"],
            "Too Short": values["too_short"],
            "Missing Section": values["missing"],
            "Total Removed": total,
            "Too Long %": round(values["too_long"] / total * 100, 2) if total > 0 else 0.0,
            "Too Short %": round(values["too_short"] / total * 100, 2) if total > 0 else 0.0,
            "Missing Section %": round(values["missing"] / total * 100, 2) if total > 0 else 0.0,
            "Removed % of Original": round(total / original * 100, 2) if original > 0 else 0.0,
        })

df = pd.DataFrame(rows)
df = df.sort_values(by=["Model", "Benchmark"])

# 모델별 합계
df_sum = df.groupby("Model")[["Original Count", "Too Long", "Too Short", "Missing Section", "Total Removed"]].sum().reset_index()
df_sum["Too Long %"] = round(df_sum["Too Long"] / df_sum["Total Removed"] * 100, 2)
df_sum["Too Short %"] = round(df_sum["Too Short"] / df_sum["Total Removed"] * 100, 2)
df_sum["Missing Section %"] = round(df_sum["Missing Section"] / df_sum["Total Removed"] * 100, 2)
df_sum["Removed % of Original"] = round(df_sum["Total Removed"] / df_sum["Original Count"] * 100, 2)

# 벤치마크별 합계
df_bench = df.groupby("Benchmark")[[
    "Original Count", "Too Long", "Too Short", "Missing Section", "Total Removed"
]].sum().reset_index()

# 비율 계산
df_bench["Too Long %"] = (df_bench["Too Long"] / df_bench["Total Removed"] * 100).round(2)
df_bench["Too Short %"] = (df_bench["Too Short"] / df_bench["Total Removed"] * 100).round(2)
df_bench["Missing Section %"] = (df_bench["Missing Section"] / df_bench["Total Removed"] * 100).round(2)
df_bench["Removed % of Original"] = (df_bench["Total Removed"] / df_bench["Original Count"] * 100).round(2)

# 컬럼 순서 정리
cols_order_bench = [
    "Benchmark",
    "Original Count", "Total Removed", "Removed % of Original",
    "Too Long", "Too Short", "Missing Section",
    "Too Long %", "Too Short %", "Missing Section %"
]
df_bench = df_bench[cols_order_bench]

print("\n=== 벤치마크별 합계 (비율 포함) ===")
print(df_bench.to_string(index=False))

# 출력
print("\n=== Benchmark별 상세 (비율 포함) ===")
cols_order_detail = [
    "Model", "Benchmark",
    "Original Count", "Total Removed", "Removed % of Original",
    "Too Long", "Too Short", "Missing Section",
    "Too Long %", "Too Short %", "Missing Section %"
]
print(df[cols_order_detail].to_string(index=False))

print("\n=== 모델별 합계 (비율 포함) ===")
cols_order_sum = [
    "Model",
    "Original Count", "Total Removed", "Removed % of Original",
    "Too Long", "Too Short", "Missing Section",
    "Too Long %", "Too Short %", "Missing Section %"
]
print(df_sum[cols_order_sum].to_string(index=False))

# %%
### 합침

import re
import pandas as pd
from collections import defaultdict

def parse_logs_with_named_benchmarks(log_text: str):
    # 로그 안에서 등장하는 순서대로 benchmark 이름을 매핑
    benchmark_names = [
        "HumanEval",
        "LiveCodeBench",
        "CodeForces",
        "CodeForces-Challenging",
    ]

    # 패턴 정의
    model_pattern = re.compile(r"Processing model:\s+(.+?)\s+={5,}")
    benchmark_pattern = re.compile(r"Found\s+(\d+)\s+problems.*each with\s+(\d+)\s+narratives")
    removed_pattern = re.compile(
        r"\(removed\s+(\d+)\s+too_long,\s+(\d+)\s+too_short,\s+(\d+)\s+missing_section\)"
    )

    stats = defaultdict(lambda: defaultdict(lambda: {
        "original": 0, "too_long": 0, "too_short": 0, "missing": 0
    }))

    current_model = None
    current_bench_idx = -1

    for line in log_text.splitlines():
        # 모델명
        m_model = model_pattern.search(line)
        if m_model:
            current_model = m_model.group(1).strip()
            current_bench_idx = -1
            continue

        # Found X problems, each with Y narratives
        m_bench = benchmark_pattern.search(line)
        if m_bench and current_model:
            current_bench_idx += 1
            num_problems, narratives_per_problem = map(int, m_bench.groups())
            bench_name = benchmark_names[current_bench_idx]
            stats[current_model][bench_name]["original"] += num_problems * narratives_per_problem
            continue

        # (removed X too_long, Y too_short, Z missing_section)
        m_removed = removed_pattern.search(line)
        if m_removed and current_model and current_bench_idx >= 0:
            tl, ts, ms = map(int, m_removed.groups())
            bench_name = benchmark_names[current_bench_idx]
            stats[current_model][bench_name]["too_long"] += tl
            stats[current_model][bench_name]["too_short"] += ts
            stats[current_model][bench_name]["missing"] += ms

    # Benchmark별 상세 DataFrame 생성
    rows = []
    for model, benchmarks in stats.items():
        for bench, values in benchmarks.items():
            total_removed = values["too_long"] + values["too_short"] + values["missing"]
            kept = values["original"] - total_removed
            rows.append({
                "Model": model,
                "Benchmark": bench,
                "Original Count": values["original"],
                "Too Long": values["too_long"],
                "Too Short": values["too_short"],
                "Missing Section": values["missing"],
                "Total Removed": total_removed,
                "Kept": kept,
                "Removed % of Original": round(total_removed / values["original"] * 100, 2) if values["original"] > 0 else 0.0,
                "Kept % of Original": round(kept / values["original"] * 100, 2) if values["original"] > 0 else 0.0
            })

    df = pd.DataFrame(rows)

    # 모델별 합계
    df_sum = df.groupby("Model")[[
        "Original Count", "Too Long", "Too Short", "Missing Section", "Total Removed", "Kept"
    ]].sum().reset_index()
    df_sum["Removed % of Original"] = (df_sum["Total Removed"] / df_sum["Original Count"] * 100).round(2)
    df_sum["Kept % of Original"] = (df_sum["Kept"] / df_sum["Original Count"] * 100).round(2)

    return df, df_sum


# 사용 예시
with open("log_output.txt", "r") as f:
    log_text = f.read()
df_detail, df_sum = parse_logs_with_named_benchmarks(log_text)
print("\n=== Benchmark별 상세 ===")
print(df_detail.to_string(index=False))
print("\n=== 모델별 합계 ===")
print(df_sum.to_string(index=False))


# %%
