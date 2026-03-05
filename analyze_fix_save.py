# %%
"""
Fix empty-code samples in algorithm_analysis jsonl files.
"code": "" 인 샘플의 analysis를 통일된 N/A 포맷으로 교체한다.
원본 파일은 변경하지 않고 algorithm_analysis_fixed/ 디렉토리에 새 파일로 저장한다.
"""

import json
import os
import glob

BASE_DIR     = "/Users/jang-geonhui/Downloads/pil_llm_download"
ANALYSIS_DIR = os.path.join(BASE_DIR, "algorithm_analysis")
FIXED_DIR    = os.path.join(BASE_DIR, "algorithm_analysis_fixed")  # 새 파일 저장 경로

NA_ANALYSIS = "---\n\n- Core Algorithm: N/A\n- Implementation Detail: No code generated.\n\n---"


def fix_file(src_path: str, dst_path: str) -> tuple[int, int]:
    """
    파일 내 빈 코드 샘플의 analysis를 NA_ANALYSIS로 교체.
    결과는 dst_path에 저장 (src_path는 변경하지 않음).
    반환: (전체 샘플 수, 수정된 샘플 수)
    """
    lines = []
    total = 0
    fixed = 0

    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            for sample in obj.get("results", []):
                total += 1
                if sample.get("code", "") == "":
                    sample["analysis"] = NA_ANALYSIS
                    fixed += 1

            lines.append(json.dumps(obj, ensure_ascii=False))

    with open(dst_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return total, fixed


if __name__ == "__main__":
    pattern = os.path.join(ANALYSIS_DIR, "algorithm_analysis_*.jsonl")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found in {ANALYSIS_DIR}")
    else:
        print(f"Found {len(files)} files.")
        print(f"Saving fixed files to: {FIXED_DIR}\n")

    os.makedirs(FIXED_DIR, exist_ok=True)

    grand_total = 0
    grand_fixed = 0

    for src_path in files:
        fname    = os.path.basename(src_path)
        dst_path = os.path.join(FIXED_DIR, fname)
        total, fixed = fix_file(src_path, dst_path)
        grand_total += total
        grand_fixed += fixed
        status = f"fixed {fixed:>4} / {total:>6} samples"
        print(f"{fname:<60}  {status}")

    print(f"\nDone.  Total fixed: {grand_fixed} / {grand_total}")
# %%
