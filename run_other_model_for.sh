MODEL_IDS=(
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "deepseek-ai/DeepSeek-V2-Lite-Chat"
    # "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

for MODEL_ID in "${MODEL_IDS[@]}"
do
    echo "==================== Running for: $MODEL_ID ===================="
    python change_coding_into_narrative_other_model_variant.py \
        --model_id "$MODEL_ID" \
        --batch_size 8 \
        --max_tokens 4096 \
        --num_samples 5 \
        --mode "narrative"
    echo "===============================================================\n"
done

for ((i=1; i<=100000; i++)); do
    echo "$i"
    python /home/work/users/PIL_ghj/LLM/code/generate_qa_datasets_copy.py
done
