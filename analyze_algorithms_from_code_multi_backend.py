# analyze_algorithms_from_code_multi_backend.py
import os
import json
import time
import argparse
from tqdm import tqdm

from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfigDict
import anthropic

# ====================
# 유틸 함수
# ====================
def load_existing_ids(path):
    """이미 저장된 question_id 불러오기"""
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(obj["question_id"])
            except Exception:
                continue
    return ids

# ====================
# 모델 호출 함수
# ====================
PROMPT_HEADER = """You are given a piece of code. Your task is to identify the single most essential algorithm used in the code and describe how it is implemented. Follow the format strictly.

First, choose the algorithm used in the given code from the following list:

- Graph Algorithms
- Dynamic Programming
- Greedy Algorithms
- Sorting and Searching
- String Algorithms
- Data Structures
- Mathematics and Number Theory
- Simulation and Implementation

Then, write one concise sentence describing the specific way the algorithm is implemented in the code.

You must follow this format exactly and do not output any extra text outside the format:

---

- Core Algorithm: (chosen algorithm)
- Implementation Detail: (concise description)

---

The code is as follows:

"""

def call_gpt(client, code):
    response = client.responses.create(
        model="gpt-4.1-mini-2025-04-14",
        input=[{"role": "user", "content": PROMPT_HEADER + "\n\n" + code}],
        temperature=0.2,
    )
    return response.output_text.strip()

def call_gemini(client, code):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT_HEADER + "\n\n" + code,
        config=GenerateContentConfigDict(
            temperature=0.2,
            candidate_count=1,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0),
        ),
    )
    if response and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    return ""

def call_claude(client, code):
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=2048,
        temperature=0.2,
        messages=[{"role": "user", "content": PROMPT_HEADER + "\n\n" + code}],
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
    parser.add_argument("--mode", type=str, required=True, choices=["narrative", "original"])
    parser.add_argument("--backend", type=str, required=True, choices=["gpt", "gemini", "claude"])
    parser.add_argument("--benchmark", type=str, required=True, choices=["humaneval_filtered", "test6", "codeforces", "codeforces_challenging"])
    args = parser.parse_args()

    backend = args.backend
    benchmark = args.benchmark

    # --- API client 초기화 ---
    if backend == "gpt":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif backend == "gemini":
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_PROJECT_ID"),
            location=os.getenv("GOOGLE_LOCATION"),
        )
    elif backend == "claude":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError("Unsupported backend")

    # ----------------
    # 입력 경로 설정
    # ----------------
    base_dir = "/home/work/users/PIL_ghj/LLM/code/LiveCodeBench"

    if args.mode == "narrative":
        input_paths = []
        for convert_option in ["with_io", "merge"]:
            for i in range(1, 6):
                p = os.path.join(
                    base_dir,
                    f"output_search_algorithm/{benchmark}/narrative_by_gemini_search_algorithm_narrative_{i}_{convert_option}",
                    "Gemini-2.5-Flash",
                    "Scenario.codegeneration_1_0.2_eval_all.json"
                )
                input_paths.append(p)

    elif args.mode == "original":
        input_paths = [
            os.path.join(
                base_dir,
                f"output/{benchmark}/original/Gemini-2.5-Flash/Scenario.codegeneration_5_0.2_eval_all.json"
            ),
            os.path.join(
                base_dir,
                f"output_original_1/{benchmark}/original/Gemini-2.5-Flash/Scenario.codegeneration_5_0.2_eval_all.json"
            ),
        ]
    
    output_file = os.path.join(base_dir, "algorithm_analysis", f"algorithm_analysis_{benchmark}_{args.mode}_{backend}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    existing_ids = load_existing_ids(output_file)

    print(f"[Logging] Selected mode={args.mode}, backend={backend}")
    print(f"[Logging] Found {len(input_paths)} input files.")

    # ----------------
    # 데이터 로딩
    # ----------------
    all_problems = {}
    total_samples = 0
    for path in input_paths:
        print(f"[Logging] Loading file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            problems = json.load(f)
        print(f"[Logging]   Loaded {len(problems)} problems from file.")
        for p in problems:
            qid = p["question_id"]
            if qid not in all_problems:
                all_problems[qid] = {
                    "question_id": qid,
                    "question_title": p.get("question_title", ""),
                    "question_content": p.get("question_content", ""),
                    "platform": p.get("platform", ""),
                    "contest_id": p.get("contest_id", ""),
                    "difficulty": p.get("difficulty", ""),
                    "samples": []   # code + grade 같이 저장
                }
            codes_here = p.get("code_list", [])
            grades_here = p.get("graded_list", [None] * len(codes_here))
            for code, grade in zip(codes_here, grades_here):
                all_problems[qid]["samples"].append({"code": code, "is_correct": grade})
            total_samples += len(codes_here)

        print(f"[Logging]   Total samples so far: {total_samples}")

    print(f"[Logging] Loaded {len(all_problems)} unique problems in total.")
    print(f"[Logging] Total code samples collected: {total_samples}")

    # ----------------
    # 결과 생성 및 저장
    # ----------------
    with open(output_file, "a", encoding="utf-8") as outfile:
        for qid, prob in tqdm(all_problems.items(), desc="Processing problems", unit="problem"):
            if qid in existing_ids:
                print(f"[Skip] Already processed {qid}")
                continue

            samples = prob["samples"]
            # print(f"[Processing] qid={qid} with {len(samples)} code samples.")

            results = []
            for idx, sample in enumerate(samples):
                code = sample["code"]
                grade = sample["is_correct"]
                try:
                    if backend == "gpt":
                        analysis = call_gpt(client, code)
                    elif backend == "gemini":
                        analysis = call_gemini(client, code)
                    elif backend == "claude":
                        analysis = call_claude(client, code)
                except Exception as e:
                    print(f"[Error] qid={qid}, sample={idx}: {e}")
                    analysis = ""
                results.append({
                    "code": code,
                    "analysis": analysis,
                    "is_correct": grade
                })
                time.sleep(1)

            out_obj = {
                "question_id": qid,
                "question_title": prob["question_title"],
                "question_content": prob["question_content"],
                "platform": prob["platform"],
                "contest_id": prob["contest_id"],
                "difficulty": prob["difficulty"],
                "n_codes": len(samples),
                "results": results
            }
            outfile.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            outfile.flush()
            # print(f"[Done] qid={qid}, saved {len(samples)} samples.")
