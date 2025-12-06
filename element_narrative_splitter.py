# %%
import os
import json
import re

def remove_algorithm_and_genre(text: str) -> str:
    """
    Removes the Algorithm Category and Narrative Genre sections (case-insensitive).
    Handles variations with/without leading '-' and with/without ':'.
    """
    # Algorithm Category 부분 제거
    text = re.sub(
        r"(?si)^\s*(-\s*)?Algorithm Category:?\s*.*?(?=\n\s*(-\s*)?Narrative Genre:?\s*|\Z)",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Narrative Genre 부분 제거
    text = re.sub(
        r"(?si)^\s*(-\s*)?Narrative Genre:?\s*.*?(?=\n\s*(-\s*)?(Task Overview|Constraints|Example Input/Output)|\Z)",
        "",
        text,
        flags=re.MULTILINE,
    )

    # 남은 문자열 앞뒤 공백 정리
    return text.strip()


if __name__ == "__main__":
    
    # === 옵션 직접 지정 ===
    option = "sot_template"  # "search" → 그대로 유지, "search_algorithm" → 알고리즘/장르 제거 / "search_algorithm_mismatch" "sot_template"
    backend = "gemini"  # ["gemini", "chatgpt", "claude"]
    
    input_jsonl_path_list = [
        ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", f"humaneval_filtered_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", f"test6_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", f"codeforces_narrative_by_{backend}_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", f"codeforces_mid_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", f"codeforces_challenging_narrative_by_{backend}_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_longer_in_lcb_format.jsonl", "CodeForces", f"codeforces_longer_narrative_by_{backend}_{option}.jsonl"),
    ]
    base_input_dir = f"/home/work/users/PIL_ghj/LLM/datasets/{backend}_{option}"
    
    for _, output_path_name, file_name in input_jsonl_path_list:
        input_jsonl_path = os.path.join(base_input_dir, output_path_name, file_name)
        base_output_path = input_jsonl_path.replace(".jsonl", "")

        # 전체 파일 로드
        with open(input_jsonl_path, "r", encoding="utf-8") as infile:
            problems = [json.loads(line) for line in infile]

        # narratives 개수 확인 (첫 문제 기준)
        if problems and "narratives" in problems[0]:
            num_variants = len(problems[0]["narratives"])
        else:
            raise ValueError("No narratives found in the input file.")

        print(f"[Logging] Found {len(problems)} problems, each with {num_variants} narratives.")

        # 각 narrative별 jsonl 파일 생성
        for variant_idx in range(num_variants):
            output_jsonl_path = f"{base_output_path}_narrative_{variant_idx+1}.jsonl"

            if os.path.exists(output_jsonl_path):
                print(f"[Logging] Removing existing file: {output_jsonl_path}")
                os.remove(output_jsonl_path)

            with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
                for problem in problems:
                    new_problem = dict(problem)  # shallow copy
                    narratives = new_problem.get("narratives", [])
                    if variant_idx < len(narratives):
                        content = narratives[variant_idx]
                        if option == "search_algorithm":  # 알고리즘/장르 제거 모드
                            content = remove_algorithm_and_genre(content)
                        new_problem["question_content"] = content
                    else:
                        new_problem["question_content"] = ""  # 혹시 개수가 안 맞을 경우 대비

                    # 필요없는 key 제거
                    new_problem.pop("narratives", None)
                    new_problem.pop("random_combinations", None)  # 혹시 있으면 제거

                    outfile.write(json.dumps(new_problem, ensure_ascii=False) + "\n")

            print(f"[Logging] Saved {output_jsonl_path}")

        print(f"[Logging] Finished generating {num_variants} narrative files.")

# %%
