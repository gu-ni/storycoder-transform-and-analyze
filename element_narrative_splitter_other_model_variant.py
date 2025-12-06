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


def filter_paraphrase(model_id, text, max_tokens=5120, min_tokens=10):
    """
    내러티브 필터링:
    - max_tokens 도달 → 제외
    - min_tokens 이하 → 제외
    """
    tokenizer = get_tokenizer(model_id)
    num_tokens = len(tokenizer.encode(text))

    if num_tokens >= max_tokens:
        return None, num_tokens, "too_long"
    if num_tokens <= min_tokens:
        return None, num_tokens, "too_short"

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
    # "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "deepseek-ai/DeepSeek-V2-Lite-Chat",
    # "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    # "Qwen/Qwen2.5-7B-Instruct",
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
    mode = "narrative"  # "narrative", "paraphrase"

    input_jsonl_path_list = [
        # ("HumanEval", f"humaneval_filtered_{mode}_by_llm.jsonl"),
        # ("LiveCodeBench", f"test6_{mode}_by_llm.jsonl"),
        # ("CodeForces", f"codeforces_{mode}_by_llm.jsonl"),
        # ("CodeForces", f"codeforces_challenging_{mode}_by_llm.jsonl"),
        ("CodeForces", f"codeforces_longer_{mode}_by_llm.jsonl"),
    ]

    for model_id in TARGET_MODEL_IDS:
        model_name = MODEL_TEMPLATES[model_id]["name"]
        base_input_dir = f"/home/work/users/PIL_ghj/LLM/datasets/Ablation/diff_quality_{option}/{model_name}"

        print(f"\n==================== Processing model: {model_name} ====================")

        for output_path_name, file_name in input_jsonl_path_list:
            input_jsonl_path = os.path.join(base_input_dir, output_path_name, file_name)
            base_output_path = input_jsonl_path.replace(".jsonl", "")

            if not os.path.exists(input_jsonl_path):
                print(f"[Warning] File not found: {input_jsonl_path}")
                continue

            with open(input_jsonl_path, "r", encoding="utf-8") as infile:
                problems = [json.loads(line) for line in infile]

            if problems and f"{mode}s" in problems[0]:
                num_variants = len(problems[0][f"{mode}s"])
            else:
                print(f"[Error] No {mode}s found in {input_jsonl_path}")
                continue

            print(f"[Logging] Found {len(problems)} problems, each with {num_variants} {mode}s.")

            for variant_idx in range(num_variants):
                output_jsonl_path = f"{base_output_path}_{mode}_{variant_idx+1}.jsonl"

                if os.path.exists(output_jsonl_path):
                    os.remove(output_jsonl_path)

                removed_count = {"too_long": 0, "too_short": 0, "missing_section": 0}

                with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
                    for problem in problems:
                        new_problem = dict(problem)
                        variants = new_problem.get(f"{mode}s", [])

                        if variant_idx < len(variants):
                            content = variants[variant_idx]

                            if option == "search_algorithm" and mode == "narrative":
                                content = remove_algorithm_and_genre(content)
                                
                            
                            if mode == "narrative":
                                filtered, num_tokens, status = filter_narrative(model_id, content, max_tokens=4055, min_tokens=15)
                            elif mode == "paraphrase":
                                filtered, num_tokens, status = filter_paraphrase(model_id, content, max_tokens=4055, min_tokens=15)
                            if status != "ok":
                                removed_count[status] += 1
                                continue  # 이 문제 샘플은 저장 안 함

                            new_problem["question_content"] = filtered
                        else:
                            new_problem["question_content"] = ""

                        new_problem.pop(f"{mode}s", None)
                        new_problem.pop("random_combinations", None)

                        outfile.write(json.dumps(new_problem, ensure_ascii=False) + "\n")

                print(
                    f"\n[Logging] Saved {output_jsonl_path} "
                    f"(removed {removed_count['too_long']} too_long, "
                    f"{removed_count['too_short']} too_short, "
                    f"{removed_count['missing_section']} missing_section)\n"
                )


        print(f"[Logging] Finished processing for model: {model_name}")

# %%
