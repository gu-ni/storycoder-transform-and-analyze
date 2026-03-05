# %%
import os
import json
import re

base_dir = "/Users/jang-geonhui/Downloads/pil_llm_download"
save_folder = "algorithm_edited"

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

def extract_core_algorithm(text):
    """analysis 문자열에서 Core Algorithm 부분만 추출"""
    if not text:
        return ""
    m = re.search(r"Core Algorithm:\s*(.+)", text)
    return m.group(1).strip() if m else ""


# =======================
# 1단계: Core Algorithm만 추출해 편집용 파일 저장 (+ dict_index 추가)
# =======================
def export_core_algorithms(benchmark_name, mode, backend="gemini"):
    A_path = os.path.join(
        base_dir,
        f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}.jsonl"
    )
    os.makedirs(os.path.join(base_dir, save_folder), exist_ok=True)

    out_path = os.path.join(
        base_dir,
        f"{save_folder}/core_algos_edit_{benchmark_name}_{mode}_{backend}.jsonl"
    )

    # 이미 편집 파일이 존재하면 중단
    if os.path.exists(out_path):
        print(f"[Warning] File already exists: {out_path}")
        print("[Aborted] Export skipped to avoid overwriting.")
        return

    with open(A_path, "r") as fin, open(out_path, "w") as fout:
        for dict_index, line in enumerate(fin):
            obj = json.loads(line)
            qid = obj["question_id"]

            for idx, s in enumerate(obj["results"]):
                algo = extract_core_algorithm(s.get("analysis", ""))
                edit_entry = {
                    "dict_index": dict_index,  # jsonl 내 dict 위치
                    "question_id": qid,
                    "index": idx,  # results 리스트 안에서의 위치
                    "core_algorithm": algo
                }
                fout.write(json.dumps(edit_entry, ensure_ascii=False) + "\n")

    print(f"[Exported] Core algorithms → {out_path}")


# =======================
# 2단계: ALLOWED_ALGOS에 없는 항목 확인
# =======================
def check_invalid_algorithms(benchmark_name, mode, backend="gemini"):
    edit_path = os.path.join(
        base_dir,
        f"{save_folder}/core_algos_edit_{benchmark_name}_{mode}_{backend}.jsonl"
    )

    invalid_entries = []
    with open(edit_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            algo = obj["core_algorithm"]
            if algo and algo not in ALLOWED_ALGOS:
                content = f"{obj['dict_index']}-{obj['index']} | {obj['question_id']} | {obj['core_algorithm']}"
                invalid_entries.append(content)

    if invalid_entries:
        print(f"[Invalid] Found {len(invalid_entries)} entries not in ALLOWED_ALGOS:")
        for e in invalid_entries[:50]:  # 앞부분만 샘플로 출력
            print(e)
        if len(invalid_entries) > 50:
            print(f"... (and {len(invalid_entries)-50} more)")
    else:
        print("[OK] All core_algorithm entries are within ALLOWED_ALGOS.")

    return invalid_entries


# =======================
# 2단계: 편집된 파일을 반영해 새로운 A' 파일 생성
# =======================
def import_core_algorithms(benchmark_name, mode, backend="gemini"):
    A_path = os.path.join(
        base_dir,
        f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}.jsonl"
    )
    edit_path = os.path.join(
        base_dir,
        f"{save_folder}/core_algos_edit_{benchmark_name}_{mode}_{backend}.jsonl"
    )
    out_path = os.path.join(
        base_dir,
        f"algorithm_analysis/algorithm_analysis_{benchmark_name}_{mode}_{backend}_edited.jsonl"
    )

    # 기존 파일 있으면 삭제
    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"[Info] 기존 파일 삭제 → {out_path}")

    # (1) 편집본 로딩 → dict[(qid, idx)] = core_algorithm
    edit_map = {}
    with open(edit_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            edit_map[(obj["question_id"], obj["index"])] = obj["core_algorithm"]

    # (2) 원본 로딩 후 교체
    with open(A_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            qid = obj["question_id"]

            for idx, s in enumerate(obj["results"]):
                key = (qid, idx)
                if key in edit_map:
                    algo = edit_map[key]
                    # analysis 내 Core Algorithm 교체
                    analysis = s.get("analysis", "")
                    if "Core Algorithm:" in analysis:
                        analysis = re.sub(
                            r"Core Algorithm:\s*.+",
                            f"Core Algorithm: {algo}",
                            analysis
                        )
                    else:
                        # 만약 없는 경우 그냥 추가
                        analysis = (analysis + "\nCore Algorithm: " + algo).strip()
                    s["analysis"] = analysis
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[Updated] Edited algorithms 반영 → {out_path}")


# %%
# =======================
# 사용 예시
# =======================
export_core_algorithms("codeforces_challenging", "narrative", backend="gemini")

# %%
invalids = check_invalid_algorithms("test6", "original", backend="gemini")
# %%
import_core_algorithms("test6", "narrative", backend="gemini")
# %%
