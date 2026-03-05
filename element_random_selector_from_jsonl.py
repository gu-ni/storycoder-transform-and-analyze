# element_random_selector_from_jsonl.py
import os
import re
import json
import random


def extract_element(text: str, element_name: str) -> str:
    """
    Extracts the content of a specific element from a narrative string.
    Only the three allowed headers are recognized:
    - Task Overview
    - Constraints
    - Example Input/Output
    """
    pattern = (
        rf"(?m)^\s*- {element_name}\s*(.*?)(?=^\s*-(Task Overview|Constraints|Example Input/Output)|\Z)"
    )
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def make_random_combinations(task_variants, constraint_variants, example_variants, num_combos=5):
    """
    Generate random narrative combinations by sampling one element
    from each category (task, constraints, example).
    """
    combos = []
    for _ in range(num_combos):
        t = random.choice(task_variants) if task_variants else ""
        c = random.choice(constraint_variants) if constraint_variants else ""
        e = random.choice(example_variants) if example_variants else ""
        combined = f"- Task Overview\n{t}\n\n- Constraints\n{c}\n\n- Example Input/Output\n{e}"
        combos.append(combined)
    return combos


if __name__ == "__main__":

    input_jsonl_path_list = [
        # ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", "humaneval_filtered_arrative_by_gemini_search.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", "test6_arrative_by_gemini_search.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", "codeforces_arrative_by_gemini_search.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", "codeforces_mid_arrative_by_gemini_search.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", "codeforces_challenging_narrative_by_gemini_search.jsonl"),
    ]
    
    base_input_dir = "/home/work/users/PIL_ghj/LLM/datasets/gemini_search"
    base_output_dir = "/home/work/users/PIL_ghj/LLM/datasets/gemini_random"

    for _, output_dir_name, output_file_name in input_jsonl_path_list:
        try:
            input_jsonl_path = os.path.join(base_input_dir, output_dir_name, output_file_name)
            output_path = os.path.join(base_output_dir, output_dir_name)
            os.makedirs(output_path, exist_ok=True)
            output_jsonl_path = os.path.join(output_path, output_file_name)
            
            if os.path.exists(output_jsonl_path):
                print(f"[Logging] Removing existing file: {output_jsonl_path}")
                os.remove(output_jsonl_path)
                
            with open(input_jsonl_path, "r", encoding="utf-8") as infile, \
                 open(output_jsonl_path, "w", encoding="utf-8") as outfile:

                for i, line in enumerate(infile):
                    try:
                        problem = json.loads(line)
                        qid = problem.get("question_id")
                        print(f"\n[Logging] {output_dir_name} | Starting {i}-th Problem (qid={qid})...")

                        narratives = problem.get("narratives", [])

                        # Extract per-element variants
                        task_variants = [extract_element(n, "Task Overview") for n in narratives]
                        constraint_variants = [extract_element(n, "Constraints") for n in narratives]
                        example_variants = [extract_element(n, "Example Input/Output") for n in narratives]

                        # Generate random 5 combinations
                        random_combos = make_random_combinations(task_variants, constraint_variants, example_variants, num_combos=5)

                        # Save
                        problem["random_combinations"] = random_combos
                        outfile.write(json.dumps(problem, ensure_ascii=False) + "\n")

                        print(f"[Logging] Finished problem {qid}")

                    except Exception as e:
                        print(f"[Error] Processing idx={i}, qid={qid}: {e}")
                        continue

            print(f"[Logging] Finished dataset: {input_jsonl_path}")

        except Exception as e:
            print(f"[Error] Processing dataset {input_jsonl_path}: {e}")
            continue
