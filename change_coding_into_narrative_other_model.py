import os
import time
import json
import argparse
import gc
import torch
import sys

from vllm import LLM, SamplingParams
from instruction_template import INSTRUCTION_HUMANEVAL, INSTRUCTION_INCLUDING_HINTS, genres


MODEL_TEMPLATES = {
    "deepseek-ai/deepseek-coder-6.7b-instruct": {
        "name": "DSCoder-6.7b-Ins",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": { # error
        "name": "DSCoder-V2-Lite-Instruct",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "bigcode/starcoder2-15b": { # error
        "name": "StarCoder2-15b",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": { # OK
        "name": "LLama3.1-8b-Ins",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "google/gemma-2-9b-it": { # OK
        "name": "Gemma-2-9b-Ins",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "google/gemma-2-27b-it": { # OK
        "name": "Gemma-2-27b-Ins",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": { # OK
        "name": "Qwen2.5-Coder-Ins-7B",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "Qwen/Qwen2.5-Coder-32B-Instruct": {
        "name": "Qwen2.5-Coder-Ins-32B",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "name": "Mistral-Small-24B-Instruct-2501",
        "system_prompt": "You are an imaginative storyteller who follows instructions well.",
    },
}


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


def load_model(model_id, log_func=print):
    """
    vLLM 모델 로드: trust_remote_code=False → 실패 시 True로 fallback
    """
    try:
        log_func(f"[Logging] Loading model without trust_remote_code: {model_id}")
        model = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=False,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
        )
        return model
    except Exception as e:
        if "trust_remote_code" in str(e):
            log_func(f"[Warning] Reloading with trust_remote_code=True for {model_id}")
            try:
                model = LLM(
                    model=model_id,
                    dtype="bfloat16",
                    tensor_parallel_size=torch.cuda.device_count(),
                    trust_remote_code=True,
                    max_model_len=4096,
                    gpu_memory_utilization=0.95,
                )
                return model
            except Exception as e2:
                log_func(f"[Error] Failed with trust_remote_code=True: {e2}")
                raise
        else:
            log_func(f"[Error] Model loading failed: {e}")
            raise


def main(args):
    model_id = args.model_id
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    if model_id not in MODEL_TEMPLATES:
        raise ValueError(f"Unsupported model_id: {model_id}")

    model_info = MODEL_TEMPLATES[model_id]
    model_name = model_info["name"]
    system_prompt = model_info["system_prompt"]

    model = load_model(model_id)

    input_jsonl_path_list = [
        "/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl",
        "/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl",
        "/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl",
        "/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl",
        "/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl",
    ]

    output_path_list = [
        f"/home/work/users/PIL_ghj/LLM/datasets/Ablation/diff_quality_including_hints/{model_name}/HumanEval",
        f"/home/work/users/PIL_ghj/LLM/datasets/Ablation/diff_quality_including_hints/{model_name}/LiveCodeBench",
        f"/home/work/users/PIL_ghj/LLM/datasets/Ablation/diff_quality_including_hints/{model_name}/CodeForces",
        f"/home/work/users/PIL_ghj/LLM/datasets/Ablation/diff_quality_including_hints/{model_name}/CodeForces",
        f"/home/work/users/PIL_ghj/LLM/datasets/Ablation/diff_quality_including_hints/{model_name}/CodeForces",
    ]

    output_jsonl_name_list = [
        "humaneval_filtered_narrative_by_llm.jsonl",
        "test6_narrative_by_llm.jsonl",
        "codeforces_narrative_by_llm.jsonl",
        "codeforces_mid_narrative_by_llm.jsonl",
        "codeforces_challenging_narrative_by_llm.jsonl",
    ]

    for input_jsonl_path, output_path, output_jsonl_name in zip(input_jsonl_path_list, output_path_list, output_jsonl_name_list):
        try:
            os.makedirs(output_path, exist_ok=True)
            output_jsonl_path = os.path.join(output_path, output_jsonl_name)
            log_path = os.path.join(output_path, output_jsonl_name + ".log")
            input_jsonl_name = input_jsonl_path.split("/")[-1]

            with open(log_path, "a", encoding="utf-8") as log_file, \
                 open(input_jsonl_path, "r", encoding="utf-8") as infile, \
                 open(output_jsonl_path, "a", encoding="utf-8") as outfile:

                def log(msg):
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

                existing_ids = load_existing_question_ids(output_jsonl_path)
                log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started processing dataset: {input_jsonl_path}")
                
                buffer = []  # [(problem, prompt_text)] 형태로 저장

                sampling_params = SamplingParams(
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=max_tokens
                )

                def flush_batch():
                    nonlocal buffer
                    if not buffer:
                        return

                    current_batch = buffer
                    buffer = []

                    def try_generate(batch):
                        try:
                            prompts = [system_prompt + "\n\n" + prompt_text for (_, prompt_text) in batch]
                            results = model.generate(prompts, sampling_params)
                            return results
                        except (ValueError, RuntimeError) as e:
                            if len(batch) == 1:
                                log(f"[Error] Failed on single prompt: {e}")
                                return None
                            else:
                                log(f"[Warning] Error in batch of size {len(batch)}: {e}")
                                mid = len(batch) // 2
                                log(f"[Retry] Splitting batch into {len(batch[:mid])} and {len(batch[mid:])}")
                                return try_generate(batch[:mid]) + try_generate(batch[mid:])

                    results = try_generate(current_batch)
                    if results is None:
                        log("[Error] All retries failed for current batch.")
                        return

                    for (problem, _), result in zip(current_batch, results):
                        qid = problem.get("question_id")
                        output_text = result.outputs[0].text.strip()

                        log("\n------------------------- Model Response -------------------------\n")
                        log(output_text)
                        log("\n----------------------------------------------------------------\n")

                        problem["question_content"] = output_text
                        outfile.write(json.dumps(problem, ensure_ascii=False) + "\n")
                        outfile.flush()
                        existing_ids.add(qid)

                
                for i, line in enumerate(infile):
                    try:
                        problem = json.loads(line)
                        qid = problem.get("question_id")
                        log(f"\n[Logging] {model_name} | {input_jsonl_name} | Starting {i}-th Problem (question_id={qid})...")

                        if qid in existing_ids:
                            log(f"[Logging] Skipping already processed question_id: {qid}")
                            continue

                        # genre = genres[int(i % len(genres))]
                        # problem["genre"] = genre
                        # instruction = INSTRUCTION_HUMANEVAL.replace("{GENRE}", genre)
                        instruction = INSTRUCTION_INCLUDING_HINTS
                        input_prompt = instruction + problem["question_content"]

                        buffer.append((problem, input_prompt))

                        # 배치가 가득 차면 flush
                        if len(buffer) == batch_size:
                            flush_batch()

                    except Exception as e:
                        log(f"[Error] Processing problem idx={i} question_id={qid}: {e}")
                        continue

                # 남은 프롬프트 처리
                flush_batch()

                log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished processing dataset: {input_jsonl_path}")

        except Exception as e:
            print(f"[Error] Processing dataset {input_jsonl_path}: {e}")
            continue

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Huggingface model ID")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--max_tokens", type=int, default=3000, help="max_tokens")
    args = parser.parse_args()
    main(args)