MODEL_IDS=(
    "deepseek-ai/deepseek-coder-6.7b-instruct"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    "bigcode/starcoder2-15b"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

for MODEL_ID in "${MODEL_IDS[@]}"
do
    echo "==================== Running for: $MODEL_ID ===================="
    python change_coding_into_narrative_other_model.py \
        --model_id "$MODEL_ID" \
        --batch_size 32
    echo "===============================================================\n"
done

for ((i=1; i<=100000; i++)); do
    echo "$i"
    python /home/work/users/PIL_ghj/LLM/code/generate_qa_datasets_copy.py
done
