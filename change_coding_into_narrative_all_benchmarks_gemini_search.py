import os
import json
import time
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfigDict
from instruction_template import INSTRUCTION_THREE_COMPONENTS, INSTRUCTION_THREE_COMPONENTS_ALGORITHM, INSTRUCTION_THREE_COMPONENTS_ALGORITHM_GIVEN_GENRE, mismatch_genre


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


# Gemini 호출
def call_gemini(model, prompt):
    response = model.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfigDict(
            temperature=1.0,
            candidate_count=1,
            max_output_tokens=8192,  # 필요에 맞게 조절
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
        )
    )
    if response and response.candidates:
        return response.candidates[0].content.parts[0].text.strip()
    return ""


if __name__ == "__main__":
    model = genai.Client(
        vertexai=True,
        # project=os.getenv("GOOGLE_PROJECT_ID"),
        project="472108490113",
        # location=os.getenv("GOOGLE_LOCATION"),
        location="us-central1",
    )

    # 내러티브 생성 횟수
    N_VARIANTS = 5  # 원하는 n값으로 변경
    option = "search_algorithm_mismatch"

    input_jsonl_path_list = [
        # ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", f"humaneval_filtered_narrative_by_gemini_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", f"test6_narrative_by_gemini_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", f"codeforces_narrative_by_gemini_{option}.jsonl"),
        # ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", f"codeforces_mid_narrative_by_gemini_{option}.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", f"codeforces_challenging_narrative_by_gemini_{option}.jsonl"),
    ]

    base_output_dir = f"/home/work/users/PIL_ghj/LLM/datasets/gemini_{option}"

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

                        instruction = INSTRUCTION_THREE_COMPONENTS_ALGORITHM_GIVEN_GENRE
                        genre_index = i % len(mismatch_genre)
                        genre = mismatch_genre[genre_index]
                        instruction = instruction.replace("{GENRE}", genre)
                        input_prompt = instruction + problem["question_content"]

                        # n번 내러티브 생성
                        narratives = []
                        
                        for v in range(N_VARIANTS):
                            new_content = call_gemini(model, input_prompt)
                            print(f"\n[Variant {v+1}] ------------------------- Gemini Response -------------------------\n")
                            print(f"Genre: {genre}\n")
                            print(new_content)
                            print("\n-------------------------------------------------------------------\n")
                            if new_content:
                                narratives.append(new_content)
                            time.sleep(1)  # API rate 제한 고려

                        # 문제에 n개의 내러티브를 리스트로 저장
                        problem["narratives"] = narratives
                        problem["genre"] = genre

                        outfile.write(json.dumps(problem, ensure_ascii=False) + "\n")
                        existing_ids.add(qid)

                        print(f"\n\n{file_name}, {i}-th Problem (question_id={qid}) Done")

                    except Exception as e:
                        print(f"[Error] Processing problem idx={i} question_id={qid}: {e}")
                        continue

            print(f"[Logging] Finished dataset: {input_jsonl_path}")

        except Exception as e:
            print(f"[Error] Processing dataset {input_jsonl_path}: {e}")
            continue

        time.sleep(2)
