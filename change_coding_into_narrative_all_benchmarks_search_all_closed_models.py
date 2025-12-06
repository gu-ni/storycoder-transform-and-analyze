import os
import json
import time
import argparse
from instruction_template import INSTRUCTION_THREE_COMPONENTS, INSTRUCTION_THREE_COMPONENTS_ALGORITHM, SOT_INSTRUCTION

# --- API clients import ---
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfigDict
from openai import OpenAI
import anthropic


# ====================
# 기존 출력 파일에서 이미 처리한 ID 수집
# ====================
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


# --------------------
# Gemini
# --------------------
def call_gemini(client, prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfigDict(
            temperature=1.0,
            candidate_count=1,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0),
        ),
    )
    if response and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    return ""


# --------------------
# ChatGPT (OpenAI)
# --------------------
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


# --------------------
# Claude (Anthropic)
# --------------------
def call_claude(client, prompt):
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=8192,
        temperature=1.0,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    content = "\n".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    return content.strip()


# ====================
# Main
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True, choices=["gemini", "chatgpt", "claude"])
    parser.add_argument("--n_variants", type=int, default=5)
    parser.add_argument("--option", type=str, default="search_algorithm")
    args = parser.parse_args()

    backend = args.backend
    N_VARIANTS = args.n_variants
    option = args.option

    # -------- API Client 초기화 --------
    if backend == "gemini":
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_PROJECT_ID"),
            location=os.getenv("GOOGLE_LOCATION"),
        )
    elif backend == "chatgpt":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif backend == "claude":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError("Unsupported backend")

    # -------- Config --------
    input_jsonl_path_list = [
        ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", f"humaneval_filtered_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", f"test6_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", f"codeforces_narrative_by_{backend}_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", f"codeforces_mid_narrative_by_{backend}_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", f"codeforces_challenging_narrative_by_{backend}_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_longer_in_lcb_format.jsonl", "CodeForces", f"codeforces_longer_narrative_by_{backend}_{option}.jsonl"),
        
    ]

    base_output_dir = f"/home/work/users/PIL_ghj/LLM/datasets/{backend}_{option}"

    # -------- Loop over datasets --------
    for input_jsonl_path, output_path_name, file_name in input_jsonl_path_list:
        try:
            output_path = os.path.join(base_output_dir, output_path_name)
            os.makedirs(output_path, exist_ok=True)
            output_jsonl_path = os.path.join(output_path, file_name)

            existing_ids = load_existing_question_ids(output_jsonl_path)

            with open(input_jsonl_path, "r", encoding="utf-8") as infile, \
                 open(output_jsonl_path, "a", encoding="utf-8") as outfile:

                for i, line in enumerate(infile):
                    try:
                        problem = json.loads(line)
                        qid = problem.get("question_id")
                        print(f"\n[Logging] {output_path_name} | Starting {i}-th Problem (qid={qid})...")
                        if qid in existing_ids:
                            print(f"[Logging] Skipping already processed qid: {qid}")
                            continue

                        instruction = SOT_INSTRUCTION  # INSTRUCTION_THREE_COMPONENTS_ALGORITHM
                        input_prompt = instruction + problem["question_content"]

                        narratives = []
                        for v in range(N_VARIANTS):
                            if backend == "gemini":
                                new_content = call_gemini(client, input_prompt)
                            elif backend == "chatgpt":
                                new_content = call_gpt(client, input_prompt)
                            elif backend == "claude":
                                new_content = call_claude(client, input_prompt)

                            print(f"\n[Variant {v+1}] ------------------------- {backend.upper()} Response -------------------------\n")
                            print(new_content)
                            print("\n-------------------------------------------------------------------\n")

                            if new_content:
                                narratives.append(new_content)
                            time.sleep(1)

                        problem["narratives"] = narratives
                        outfile.write(json.dumps(problem, ensure_ascii=False) + "\n")
                        outfile.flush()
                        existing_ids.add(qid)

                        print(f"\n\n{file_name}, {i}-th Problem (qid={qid}) Done")

                    except Exception as e:
                        print(f"[Error] Processing problem idx={i} qid={qid}: {e}")
                        continue

            print(f"[Logging] Finished dataset: {input_jsonl_path}")

        except Exception as e:
            print(f"[Error] Processing dataset {input_jsonl_path}: {e}")
            continue

        time.sleep(2)
