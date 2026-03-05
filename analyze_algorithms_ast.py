# %%
"""
AST-based Structural Analysis for StoryCoder Rebuttal
======================================================
리뷰어 질문: "whether generated programs exhibit more structured
or stepwise design at the code level"

측정 지표:
  - num_top_functions   : 최상위 함수 수 (stepwise decomposition 직접 지표)
  - num_all_functions   : 전체 함수 수 (중첩 함수 포함)
  - has_helper_func     : 보조 함수가 있는지 여부 (0/1)
  - ast_depth           : AST 최대 깊이 (구조 복잡도)
  - num_nodes           : AST 전체 노드 수
  - cyclomatic_proxy    : 분기/루프 수 (if/for/while/try/...)

분석 방식:
  정답 코드(is_correct=True)만을 대상으로
  original vs narrative 비교
  CodeForces + CodeForces Challenging → 샘플 합산하여 단일 CodeForces로 처리
  유의성 검증: Mann-Whitney U test (정규성 가정 불필요)
"""

import ast
import json
import os
import statistics
from collections import defaultdict
from scipy import stats

# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────
BASE_DIR = "/Users/jang-geonhui/Downloads/pil_llm_download"
BACKEND  = "gemini"

# 단일 벤치마크 목록 (codeforces_both는 아래에서 합산)
SINGLE_BENCHMARKS = {
    "humaneval_filtered": "HumanEval",
    "test6":              "LiveCodeBench",
}
CODEFORCES_SUBS = ["codeforces", "codeforces_challenging"]

FILE_PATTERN = "algorithm_analysis_fixed/algorithm_analysis_{benchmark}_{mode}_{backend}.jsonl"


# ───────────────────────────────────────────
# AST 분석 함수
# ───────────────────────────────────────────

def _depth(node: ast.AST, d: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    return max((_depth(c, d + 1) for c in children), default=d)


BRANCH_LOOP_NODES = (
    ast.If, ast.For, ast.While, ast.Try,
    ast.ExceptHandler, ast.With, ast.Assert,
)


def analyze_code(code: str) -> dict | None:
    if not code or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    all_nodes = list(ast.walk(tree))
    top_funcs = sum(
        1 for n in ast.iter_child_nodes(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    all_funcs = sum(
        1 for n in all_nodes
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    n_classes = sum(1 for n in all_nodes if isinstance(n, ast.ClassDef))
    cyclo     = sum(1 for n in all_nodes if isinstance(n, BRANCH_LOOP_NODES))

    return {
        "num_top_functions": top_funcs,
        "num_all_functions": all_funcs,
        "num_classes":       n_classes,
        "ast_depth":         _depth(tree),
        "num_nodes":         len(all_nodes),
        "cyclomatic_proxy":  cyclo,
        "has_helper_func":   int(top_funcs >= 2 or all_funcs > top_funcs),
    }


# ───────────────────────────────────────────
# 데이터 로딩
# ───────────────────────────────────────────

def load_jsonl(benchmark: str, mode: str) -> dict:
    path = os.path.join(
        BASE_DIR,
        FILE_PATTERN.format(benchmark=benchmark, mode=mode, backend=BACKEND),
    )
    data = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            data[obj["question_id"]] = obj["results"]
    return data


# ───────────────────────────────────────────
# 통계 헬퍼
# ───────────────────────────────────────────

def mean(lst): return round(statistics.mean(lst), 3) if lst else None
def sd(lst):   return round(statistics.stdev(lst), 3) if len(lst) > 1 else None
def pct(lst):  return round(sum(lst) / len(lst) * 100, 1) if lst else None


METRIC_KEYS = [
    "num_top_functions",
    "num_all_functions",
    "has_helper_func",
    "ast_depth",
    "num_nodes",
    "cyclomatic_proxy",
]

METRIC_LABELS = {
    "num_top_functions": "Top-level functions      (stepwise)",
    "num_all_functions": "All functions incl. nested        ",
    "has_helper_func":   "Has helper func (rate %)          ",
    "ast_depth":         "AST depth                         ",
    "num_nodes":         "AST nodes                         ",
    "cyclomatic_proxy":  "Cyclomatic proxy (if/for/while/..)",
}


# ───────────────────────────────────────────
# 벤치마크별 샘플 수집
# ───────────────────────────────────────────

def collect_samples(benchmark: str) -> tuple[dict, dict]:
    """
    반환: (vals, counters)
      vals     : mode → bucket → metric → [values]
      counters : mode → {total, empty_code, parse_fail, analyzed}
    """
    orig_data = load_jsonl(benchmark, "original")
    narr_data = load_jsonl(benchmark, "narrative")
    qids = set(orig_data) & set(narr_data)

    vals = {
        "original":  {"correct": defaultdict(list), "incorrect": defaultdict(list)},
        "narrative": {"correct": defaultdict(list), "incorrect": defaultdict(list)},
    }
    counters = {
        mode: {"total": 0, "empty_code": 0, "parse_fail": 0, "analyzed": 0}
        for mode in ("original", "narrative")
    }

    for qid in qids:
        for mode, data in [("original", orig_data), ("narrative", narr_data)]:
            for sample in data[qid]:
                code = sample.get("code", "")
                counters[mode]["total"] += 1

                if not code or not code.strip():
                    counters[mode]["empty_code"] += 1
                    continue

                m = analyze_code(code)
                if m is None:
                    counters[mode]["parse_fail"] += 1
                    continue

                counters[mode]["analyzed"] += 1
                bucket = "correct" if sample.get("is_correct", False) else "incorrect"
                for k in METRIC_KEYS:
                    vals[mode][bucket][k].append(m[k])

    return vals, counters


def merge_vals(vals_list: list[dict]) -> dict:
    """여러 벤치마크의 vals를 샘플 단위로 합산 (가중 평균과 동일)"""
    merged = {
        mode: {"correct": defaultdict(list), "incorrect": defaultdict(list)}
        for mode in ("original", "narrative")
    }
    for vals in vals_list:
        for mode in ("original", "narrative"):
            for bucket in ("correct", "incorrect"):
                for k in METRIC_KEYS:
                    merged[mode][bucket][k].extend(vals[mode][bucket][k])
    return merged


def merge_counters(counters_list: list[dict]) -> dict:
    merged = {
        mode: {"total": 0, "empty_code": 0, "parse_fail": 0, "analyzed": 0}
        for mode in ("original", "narrative")
    }
    for counters in counters_list:
        for mode in ("original", "narrative"):
            for key in merged[mode]:
                merged[mode][key] += counters[mode][key]
    return merged


# ───────────────────────────────────────────
# Mann-Whitney U test
# ───────────────────────────────────────────

def mw_test(orig_vals: list, narr_vals: list) -> str:
    """
    Mann-Whitney U test (단측: narrative > original)
    반환: 유의성 기호
      ***  p < 0.001
      **   p < 0.01
      *    p < 0.05
      (ns) not significant
    """
    if len(orig_vals) < 2 or len(narr_vals) < 2:
        return "─"
    _, p = stats.mannwhitneyu(narr_vals, orig_vals, alternative="greater")
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(ns)"


# ───────────────────────────────────────────
# 출력
# ───────────────────────────────────────────

def print_table(vals, bucket):
    print(f"\n  ── {bucket.upper()} solutions ──")
    hdr = (f"  {'Metric':<45} {'Orig':>9} {'Narr':>9} {'Δ':>9}"
           f"  {'sig':>5}  {'n_orig':>7} {'n_narr':>7}")
    print(hdr)
    print(f"  {'─' * 100}")

    for k in METRIC_KEYS:
        ov = vals["original"][bucket][k]
        nv = vals["narrative"][bucket][k]
        sig = mw_test(ov, nv)

        if k == "has_helper_func":
            om  = pct(ov)
            nm  = pct(nv)
            fmt = lambda x: (f"{x:.1f}%" if x is not None else "─")
            delta = round(nm - om, 1) if (om is not None and nm is not None) else None
        else:
            om  = mean(ov)
            nm  = mean(nv)
            fmt = lambda x: (f"{x:.3f}" if x is not None else "─")
            delta = round(nm - om, 3) if (om is not None and nm is not None) else None

        dsym = (f"+{delta}" if delta is not None and delta > 0
                else str(delta) if delta is not None else "─")

        print(f"  {METRIC_LABELS[k]:<45} {fmt(om):>9} {fmt(nm):>9} {dsym:>9}"
              f"  {sig:>5}  {len(ov):>7} {len(nv):>7}")

    print(f"\n  significance: * p<0.05  ** p<0.01  *** p<0.001  (ns) not significant")
    print(f"  (Mann-Whitney U test, one-sided: narrative > original)")


# ───────────────────────────────────────────
# 메인 실행
# ───────────────────────────────────────────

all_results = {}

# 1) 단일 벤치마크
for bm, display in SINGLE_BENCHMARKS.items():
    print(f"\n{'='*70}")
    print(f"  {display}  [{bm}]")
    print(f"{'='*70}")

    try:
        vals, counters = collect_samples(bm)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        continue

    n_q = len(load_jsonl(bm, "original"))
    print(f"  Problems : {n_q}")
    for mode in ("original", "narrative"):
        c = counters[mode]
        print(f"  [{mode:>9}] total={c['total']}  empty={c['empty_code']}"
              f"  parse_fail={c['parse_fail']}  analyzed={c['analyzed']}")

    print_table(vals, "correct")
    print_table(vals, "incorrect")
    all_results[bm] = vals

# 2) CodeForces (합산)
print(f"\n{'='*70}")
print(f"  CodeForces  [codeforces + codeforces_challenging, merged]")
print(f"{'='*70}")

cf_vals_list     = []
cf_counters_list = []
cf_n_problems    = 0

for sub in CODEFORCES_SUBS:
    try:
        v, c = collect_samples(sub)
        cf_vals_list.append(v)
        cf_counters_list.append(c)
        cf_n_problems += len(load_jsonl(sub, "original"))
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")

if cf_vals_list:
    cf_vals     = merge_vals(cf_vals_list)
    cf_counters = merge_counters(cf_counters_list)

    print(f"  Problems : {cf_n_problems}  (codeforces + codeforces_challenging)")
    for mode in ("original", "narrative"):
        c = cf_counters[mode]
        print(f"  [{mode:>9}] total={c['total']}  empty={c['empty_code']}"
              f"  parse_fail={c['parse_fail']}  analyzed={c['analyzed']}")

    print_table(cf_vals, "correct")
    print_table(cf_vals, "incorrect")
    all_results["codeforces_merged"] = cf_vals


# ───────────────────────────────────────────
# JSON 저장
# ───────────────────────────────────────────

def to_serial(vals):
    out = {}
    for mode in ("original", "narrative"):
        out[mode] = {}
        for bucket in ("correct", "incorrect"):
            out[mode][bucket] = {}
            for k, v in vals[mode][bucket].items():
                if k == "has_helper_func":
                    out[mode][bucket][k] = {"rate_pct": pct(v), "n": len(v)}
                else:
                    out[mode][bucket][k] = {"mean": mean(v), "stdev": sd(v), "n": len(v)}
    return out

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_algorithms_ast_structure_results.json")
with open(out_path, "w") as f:
    json.dump({bm: to_serial(v) for bm, v in all_results.items()}, f, indent=2)

print(f"\n결과 저장: {out_path}")
# %%
