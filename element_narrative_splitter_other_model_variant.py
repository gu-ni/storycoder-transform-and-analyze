# %%
import os
import json
import re

from transformers import AutoTokenizer

# 모델별 토크나이저 캐싱
tokenizers = {}

def get_tokenizer(model_id):
    if model_id not in tokenizers:
        tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    return tokenizers[model_id]


def filter_narrative(model_id, text, max_tokens=5120, min_tokens=10):
    """
    내러티브 필터링:
    - max_tokens 도달 → 제외
    - min_tokens 이하 → 제외
    - Task Overview / Constraints / Example Input/Output 중 하나라도 없으면 제외 (대소문자 무시)
    """
    tokenizer = get_tokenizer(model_id)
    num_tokens = len(tokenizer.encode(text))

    if num_tokens >= max_tokens:
        return None, num_tokens, "too_long"
    if num_tokens <= min_tokens:
        return None, num_tokens, "too_short"

    # 대소문자 무시 확인
    lowered = text.lower()
    required_keywords = ["task overview", "constraints", "example input/output"]
    if not all(kw in lowered for kw in required_keywords):
        return None, num_tokens, "missing_section"

    return text, num_tokens, "ok"



MODEL_TEMPLATES = {
    "deepseek-ai/deepseek-coder-6.7b-instruct": {
        "name": "DSCoder-6.7b-Ins",
    },
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": {
        "name": "DSCoder-V2-Lite-Instruct",
    },
    "bigcode/starcoder2-15b": {
        "name": "StarCoder2-15b",
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "name": "LLama3.1-8b-Ins",
    },
    "google/gemma-2-9b-it": {
        "name": "Gemma-2-9b-Ins",
    },
    "google/gemma-2-27b-it": {
        "name": "Gemma-2-27b-Ins",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "name": "Qwen2.5-Coder-Ins-7B",
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "name": "Qwen2.5-Coder-Ins-32B",
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "name": "Mistral-Small-24B-Instruct-2501",
    },
    "deepseek-ai/DeepSeek-V2-Lite-Chat": {
        "name": "DeepSeek-V2-Lite-Chat",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "name": "Qwen2.5-7B-Ins",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "name": "Qwen2.5-32B-Ins",
    },
}

# bash 파일에서 돌린 모델 ID만 선택
TARGET_MODEL_IDS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]


def remove_algorithm_and_genre(text: str) -> str:
    text = re.sub(
        r"(?si)^\s*(-\s*)?Algorithm Category:?\s*.*?(?=\n\s*(-\s*)?Narrative Genre:?\s*|\Z)",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?si)^\s*(-\s*)?Narrative Genre:?\s*.*?(?=\n\s*(-\s*)?(Task Overview|Constraints|Example Input/Output)|\Z)",
        "",
        text,
        flags=re.MULTILINE,
    )
    return text.strip()


if __name__ == "__main__":
    option = "search_algorithm"   # 또는 "search"

    input_jsonl_path_list = [
        ("HumanEval", "humaneval_filtered_narrative_by_llm.jsonl"),
        ("LiveCodeBench", "test6_narrative_by_llm.jsonl"),
        ("CodeForces", "codeforces_narrative_by_llm.jsonl"),
        ("CodeForces", "codeforces_challenging_narrative_by_llm.jsonl"),
    ]

    for model_id in TARGET_MODEL_IDS:
        model_name = MODEL_TEMPLATES[model_id]["name"]
        base_input_dir = f"/Users/jang-geonhui/Downloads/pil_llm_download/Ablation/diff_quality_{option}/{model_name}"

        print(f"\n==================== Processing model: {model_name} ====================")

        for output_path_name, file_name in input_jsonl_path_list:
            input_jsonl_path = os.path.join(base_input_dir, output_path_name, file_name)
            # base_output_path = input_jsonl_path.replace(".jsonl", "")

            if not os.path.exists(input_jsonl_path):
                print(f"[Warning] File not found: {input_jsonl_path}")
                continue

            with open(input_jsonl_path, "r", encoding="utf-8") as infile:
                problems = [json.loads(line) for line in infile]

            if problems and "narratives" in problems[0]:
                num_variants = len(problems[0]["narratives"])
            else:
                print(f"[Error] No narratives found in {input_jsonl_path}")
                continue

            print(f"[Logging] Found {len(problems)} problems, each with {num_variants} narratives.")

            for variant_idx in range(num_variants):
                # output_jsonl_path = f"{base_output_path}_narrative_{variant_idx+1}.jsonl"

                # if os.path.exists(output_jsonl_path):
                #     os.remove(output_jsonl_path)

                removed_count = {"too_long": 0, "too_short": 0, "missing_section": 0}

                # with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
                for problem in problems:
                    new_problem = dict(problem)
                    narratives = new_problem.get("narratives", [])

                    if variant_idx < len(narratives):
                        content = narratives[variant_idx]

                        if option == "search_algorithm":
                            content = remove_algorithm_and_genre(content)

                        filtered, num_tokens, status = filter_narrative(model_id, content, max_tokens=3000, min_tokens=50)
                        if status != "ok":
                            removed_count[status] += 1
                            continue  # 이 문제 샘플은 저장 안 함

                        new_problem["question_content"] = filtered
                    else:
                        new_problem["question_content"] = ""

                    new_problem.pop("narratives", None)
                    new_problem.pop("random_combinations", None)

                        # outfile.write(json.dumps(new_problem, ensure_ascii=False) + "\n")

                print(
                    # f"\n[Logging] Saved {output_jsonl_path} "
                    f"(removed {removed_count['too_long']} too_long, "
                    f"{removed_count['too_short']} too_short, "
                    f"{removed_count['missing_section']} missing_section)\n"
                )


        print(f"[Logging] Finished processing for model: {model_name}")

# %%
import re
import pandas as pd

def parse_logs(log_text: str):
    model_pattern = re.compile(r"Processing model:\s+(.+?)\s+={5,}")
    removed_pattern = re.compile(
        r"\(removed\s+(\d+)\s+too_long,\s+(\d+)\s+too_short,\s+(\d+)\s+missing_section\)"
    )
    problems_pattern = re.compile(r"Found\s+(\d+)\s+problems,\s+each with\s+(\d+)\s+narratives")

    data = {}
    current_model = None
    current_total_problems = 0
    current_narratives = 0

    for line in log_text.splitlines():
        m_model = model_pattern.search(line)
        if m_model:
            current_model = m_model.group(1).strip()
            if current_model not in data:
                data[current_model] = {
                    "too_long": 0,
                    "too_short": 0,
                    "missing_section": 0,
                    "problems": 0,
                    "narratives_per_problem": 0,
                }
            continue

        m_probs = problems_pattern.search(line)
        if m_probs:
            current_total_problems = int(m_probs.group(1))
            current_narratives = int(m_probs.group(2))
            data[current_model]["problems"] += current_total_problems
            data[current_model]["narratives_per_problem"] = current_narratives
            continue

        m_removed = removed_pattern.search(line)
        if m_removed and current_model:
            tl, ts, ms = map(int, m_removed.groups())
            data[current_model]["too_long"] += tl
            data[current_model]["too_short"] += ts
            data[current_model]["missing_section"] += ms

    rows = []
    for model, stats in data.items():
        total_slots = stats["problems"] * stats["narratives_per_problem"]
        removed = stats["too_long"] + stats["too_short"] + stats["missing_section"]
        kept = total_slots - removed
        rows.append({
            "Model": model,
            "Total Slots": total_slots,
            "Removed (too_long)": stats["too_long"],
            "Removed (too_short)": stats["too_short"],
            "Removed (missing_section)": stats["missing_section"],
            "Total Removed": removed,
            "Kept": kept,
            "Pass Rate (%)": round(kept / total_slots * 100, 2)
        })

    return pd.DataFrame(rows)

# 사용 예시:
with open("log_output.txt", "r") as f:
    log_text = f.read()
df = parse_logs(log_text)
print(df.to_string(index=False))

# %%
