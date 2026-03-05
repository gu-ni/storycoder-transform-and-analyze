import os
import json
import time
from openai import OpenAI
from instruction_template import INSTRUCTION_CLARIFY

# 기존 출력 파일에서 이미 처리한 ID 수집
def load_existing_question_ids(path):
    if not os.path.exists(path):
        return set()
    existing_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                qid = obj.get("question_id")
                if qid:
                    existing_ids.add(qid)
            except Exception:
                continue
    return existing_ids

# GPT 호출
def call_gpt(client, prompt):
    response = client.responses.create(
        model="gpt-4.1-mini-2025-04-14",
        input=[
            {
                "role": "developer",
                "content": "You are an imaginative storyteller who follows instructions well.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=1.0,
    )
    return response.output_text.strip()


if __name__ == "__main__":
    client = OpenAI()

    # === 여러 벤치마크를 한 번에 처리 ===
    input_jsonl_path_list = [
        ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", "humaneval_filtered_narrative_by_gpt_clarify.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", "test6_narrative_by_gpt_clarify.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", "codeforces_narrative_by_gpt_clarify.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", "codeforces_mid_narrative_by_gpt_clarify.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", "codeforces_challenging_narrative_by_gpt_clarify.jsonl"),
    ]

    base_output_dir = "/home/work/users/PIL_ghj/LLM/datasets/ChatGPT_clarify"

    for input_jsonl_path, output_path_name, file_name in input_jsonl_path_list:
        try:
            output_path = os.path.join(base_output_dir, output_path_name)
            os.makedirs(output_path, exist_ok=True)
            output_jsonl_path = os.path.join(output_path, file_name)

            # 중복 체크
            existing_ids = load_existing_question_ids(output_jsonl_path)

            with open(input_jsonl_path, "r", encoding="utf-8") as infile, \
                 open(output_jsonl_path, "a", encoding="utf-8") as outfile:

                for i, line in enumerate(infile):
                    try:
                        problem = json.loads(line)
                        qid = problem.get("question_id")
                        print(f"\n[Logging] {output_path_name} | Starting {i}-th Problem (question_id={qid})...")
                        if qid in existing_ids:
                            print(f"[Logging] Skipping already processed question_id: {qid}")
                            continue

                        instruction = INSTRUCTION_CLARIFY

                        input_prompt = instruction + problem["question_content"]

                        new_content = call_gpt(client, input_prompt)
                        print("\n------------------------- GPT Response -------------------------\n")
                        print(new_content)
                        print(f"\n\n{file_name}, {i}-th Problem (question_id={qid}) Done")
                        print("\n----------------------------------------------------------------\n")

                        problem["question_content"] = new_content
                        outfile.write(json.dumps(problem, ensure_ascii=False) + "\n")
                        existing_ids.add(qid)

                    except Exception as e:
                        print(f"[Error] Processing problem idx={i} question_id={qid}: {e}")
                        continue

                    time.sleep(1)

            print(f"[Logging] Finished dataset: {input_jsonl_path}")

        except Exception as e:
            print(f"[Error] Processing dataset {input_jsonl_path}: {e}")
            continue
        
        time.sleep(5)