# %%
import os
import json
import re
import random


def split_narrative_sections(text: str) -> dict:
    """
    Split a narrative text into sections: Task Overview, Constraints, Example Input/Output.
    Returns a dict with keys: overview, constraints, examples
    """
    sections = {"overview": "", "constraints": "", "examples": ""}
    # Task Overview
    m1 = re.search(r"(?si)-?\s*Task Overview:?\s*(.*?)(?=\n\s*-?\s*(Constraints|Example Input/Output)|\Z)", text)
    if m1:
        sections["overview"] = m1.group(1).strip()
    # Constraints
    m2 = re.search(r"(?si)-?\s*Constraints:?\s*(.*?)(?=\n\s*-?\s*Example Input/Output|\Z)", text)
    if m2:
        sections["constraints"] = m2.group(1).strip()
    # Example Input/Output
    m3 = re.search(r"(?si)-?\s*Example Input/Output:?\s*(.*)", text)
    if m3:
        sections["examples"] = m3.group(1).strip()
    return sections


def build_narrative(overview: str, constraints: str, examples: str) -> str:
    """
    Build a narrative text from its three components.
    """
    return (
        f"- Task Overview:\n{overview}\n\n"
        f"- Constraints:\n{constraints}\n\n"
        f"- Example Input/Output:\n{examples}"
    )


if __name__ == "__main__":
    option = "search_algorithm"
    backend = "gemini"

    input_jsonl_path_list = [
        # ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", f"humaneval_filtered_narrative_by_{backend}_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", f"test6_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", f"codeforces_narrative_by_{backend}_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", f"codeforces_mid_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", f"codeforces_challenging_narrative_by_{backend}_{option}.jsonl"),
    ]
    base_input_dir = f"/home/work/users/PIL_ghj/LLM/datasets/{backend}_{option}"

    for _, output_path_name, file_name in input_jsonl_path_list:
        input_jsonl_path = os.path.join(base_input_dir, output_path_name, file_name)

        with open(input_jsonl_path, "r", encoding="utf-8") as infile:
            problems = [json.loads(line) for line in infile]

        if problems and "narratives" in problems[0]:
            num_variants = len(problems[0]["narratives"])
        else:
            raise ValueError("No narratives found in the input file.")

        print(f"[Logging] Found {len(problems)} problems, each with {num_variants} narratives.")

        base_output_path = input_jsonl_path.replace(".jsonl", "")

        # 각 문제마다 혼합 내러티브 생성
        mixed_lists_per_problem = []  # 문제별 [10개 혼합 내러티브 리스트]
        for problem in problems:
            narratives = problem.get("narratives", [])
            if len(narratives) < 3:
                mixed_lists_per_problem.append([""] * 10)
                continue

            # 파트 분리
            parts = [split_narrative_sections(n) for n in narratives]

            # 가능한 (i,j,k) 조합 (모두 달라야 함)
            combos = [(i, j, k)
                      for i in range(len(parts))
                      for j in range(len(parts))
                      for k in range(len(parts))
                      if len({i, j, k}) == 3]

            # 랜덤 10개 선택
            if len(combos) >= 10:
                # 충분한 조합이 있으면 중복 없이 뽑기
                chosen = random.sample(combos, 10)
            else:
                # 조합이 부족하면 중복 허용해서 10개 뽑기
                chosen = random.choices(combos, k=10)

            mixed_variants = []
            for (i, j, k) in chosen:
                new_text = build_narrative(
                    parts[i]["overview"], parts[j]["constraints"], parts[k]["examples"]
                )
                mixed_variants.append(new_text)

            mixed_lists_per_problem.append(mixed_variants)

        # variant별 jsonl 파일로 저장
        for variant_idx in range(10):
            output_jsonl_path = f"{base_output_path}_mixed_narrative_{variant_idx+1}.jsonl"
            if os.path.exists(output_jsonl_path):
                os.remove(output_jsonl_path)

            with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
                for problem, mixed_variants in zip(problems, mixed_lists_per_problem):
                    new_problem = dict(problem)
                    new_problem["question_content"] = mixed_variants[variant_idx]

                    # 불필요한 키 제거
                    new_problem.pop("narratives", None)
                    new_problem.pop("random_combinations", None)

                    outfile.write(json.dumps(new_problem, ensure_ascii=False) + "\n")

            print(f"[Logging] Saved {output_jsonl_path}")

        print(f"[Logging] Finished generating 10 mixed narrative files.")
# %%
