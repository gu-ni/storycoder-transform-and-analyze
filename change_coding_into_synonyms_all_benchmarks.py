# %%
import os
import json
import time
import re
import nlpaug.augmenter.word as naw

# import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')

# === NLPaug WordNet Synonym Augmenter 설정 ===
# synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
synonym_aug = naw.WordEmbsAug(
    model_type='word2vec',
    model_path="/home/work/users/PIL_ghj/LLM/datasets/GoogleNews-vectors-negative300.bin",
    aug_p=0.2,
    top_k=2  # 유사도가 가장 높은 후보 중에서만 선택
)

# 파이썬 예약어 (치환 금지)
PYTHON_KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "break", "continue",
    "class", "import", "from", "as", "with", "try", "except", "finally",
    "in", "is", "and", "or", "not", "lambda", "yield", "pass", "global",
    "nonlocal", "assert", "del", "raise", "True", "False", "None",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", '"""', '\"\"\"', ">>>",
    "int", "float", "complex", "bool", "str", "list", "tuple", "dict", "set", "frozenset", "bytes", 
    "bytearray", "memoryview", "range", "NoneType", "Any", "Union", "Optional", "List", "Tuple", "Dict", 
    "Set", "FrozenSet", "Literal", "Callable", "TypeVar", "Generic", "Iterable", "Iterator",
    "true", "false", "none",
    
}

# 보호할 특수문자 집합 (괄호류, 따옴표, 대괄호, 중괄호 등)
END_PROTECT_CHARS = set('()[]{}"\'')

def safe_synonym_augmentation(text: str) -> str:
    protect_pattern = re.compile(
        r"\d+|[A-Za-z_]*\d[A-Za-z_]*|\b[a-zA-Z]\b|"
        + "|".join([rf"\b{re.escape(kw)}\b" for kw in PYTHON_KEYWORDS])
    )

    augmented_lines = []
    lines = text.splitlines(keepends=False)

    for line in lines:
        prefix_len = len(line) - len(line.lstrip(" \t"))
        indent = line[:prefix_len]
        content = line[prefix_len:]

        result_parts = []
        last_idx = 0

        for m in protect_pattern.finditer(content):
            start, end, block = m.start(), m.end(), m.group(0)

            # --- segment before block ---
            segment = content[last_idx:start]
            if segment.strip():
                try:
                    aug = synonym_aug.augment(segment)
                    if isinstance(aug, list):
                        aug = aug[0]
                    segment = aug
                except Exception:
                    pass

            # block과 segment 사이 공백 보정
            if result_parts and result_parts[-1]:
                if result_parts[-1][-1].isalpha() and (segment and segment[0].isalpha()):
                    result_parts[-1] += " "

            result_parts.append(segment)

            # block 추가 (앞뒤 공백 보정)
            if result_parts and result_parts[-1]:
                if result_parts[-1][-1].isalpha() and block[0].isalpha():
                    result_parts[-1] += " "
            result_parts.append(block)

            last_idx = end

        # 마지막 segment
        segment = content[last_idx:]
        if segment.strip():
            try:
                aug = synonym_aug.augment(segment)
                if isinstance(aug, list):
                    aug = aug[0]
                segment = aug
            except Exception:
                pass

        # block 끝 + segment 시작 공백 보정
        if result_parts and result_parts[-1]:
            if result_parts[-1][-1].isalpha() and (segment and segment[0].isalpha()):
                result_parts[-1] += " "

        result_parts.append(segment)

        augmented_lines.append(indent + "".join(result_parts))

    return "\n".join(augmented_lines)

# def safe_synonym_augmentation(text: str) -> str:
#     new_tokens = []

#     for m in re.finditer(r"\w+|\W+", text):
#         tok = m.group()
#         start, end = m.start(), m.end()

#         left_char = text[start - 1] if start > 0 else ""
#         right_char = text[end] if end < len(text) else ""

#         # 1. 숫자 보호
#         if re.fullmatch(r"-?\d+", tok):
#             new_tokens.append(tok)
#             continue

#         # 2. 숫자 포함된 토큰 보호
#         if any(ch.isdigit() for ch in tok):
#             new_tokens.append(tok)
#             continue

#         # 3. 예약어 보호
#         if tok in PYTHON_KEYWORDS:
#             new_tokens.append(tok)
#             continue

#         # 4. 한 글자 알파벳 보호
#         if tok.isalpha() and len(tok) == 1:
#             new_tokens.append(tok)
#             continue

#         # 5. 단어 끝이 보호 특수문자면 보호
#         if tok.isalpha():
#             if right_char in END_PROTECT_CHARS or left_char in END_PROTECT_CHARS:
#                 new_tokens.append(tok)
#                 continue

#             # 동의어 치환 시도
#             try:
#                 aug_tok = synonym_aug.augment(tok)
#                 if isinstance(aug_tok, list) and len(aug_tok) > 0:
#                     new_tokens.append(aug_tok[0])
#                 else:
#                     new_tokens.append(tok)
#             except Exception:
#                 new_tokens.append(tok)
#         else:
#             # 특수문자 그대로
#             new_tokens.append(tok)

#     return "".join(new_tokens)






# === 기존 출력 파일에서 이미 처리한 ID 수집 ===
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

if __name__ == "__main__":
    # === 여러 벤치마크를 한 번에 처리 ===
    input_jsonl_path_list = [
        ("/home/work/users/PIL_ghj/LLM/datasets/human-eval/data/HumanEval_in_lcb_format_io_filtered.jsonl", "HumanEval", "humaneval_filtered_synonym.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/live-code-bench/test6.jsonl", "LiveCodeBench", "test6_synonym.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_in_lcb_format.jsonl", "CodeForces", "codeforces_synonym.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_mid_in_lcb_format.jsonl", "CodeForces", "codeforces_mid_synonym.jsonl"),
        ("/home/work/users/PIL_ghj/LLM/datasets/codeforces/codeforces_challenging_in_lcb_format.jsonl", "CodeForces", "codeforces_challenging_synonym.jsonl"),
    ]

    base_output_dir = "/home/work/users/PIL_ghj/LLM/datasets/synonym"

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

                        original_content = problem["question_content"]

                        # 안전한 동의어 치환 적용
                        new_content = safe_synonym_augmentation(original_content)

                        print("\n------------------------- Synonym Augmentation -------------------------\n")
                        print("### Original:\n", original_content)
                        print("### Augmented:\n", new_content)
                        print(f"\n\n{file_name}, {i}-th Problem (question_id={qid}) Done")
                        print("\n-----------------------------------------------------------------------------\n")

                        problem["question_content"] = new_content
                        outfile.write(json.dumps(problem, ensure_ascii=False) + "\n")
                        existing_ids.add(qid)

                    except Exception as e:
                        print(f"[Error] Processing problem idx={i} question_id={qid}: {e}")
                        continue


            print(f"[Logging] Finished dataset: {input_jsonl_path}")

        except Exception as e:
            print(f"[Error] Processing dataset {input_jsonl_path}: {e}")
            continue
        
