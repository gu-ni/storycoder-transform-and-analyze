MODES=("narrative" "original")
BACKENDS=("gemini")
BENCHMARKS=("humaneval_filtered")

BASE_DIR="/home/work/users/PIL_ghj/LLM/code/LiveCodeBench"
OUT_DIR="${BASE_DIR}/algorithm_analysis"

for mode in "${MODES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        for benchmark in "${BENCHMARKS[@]}"; do
            output_file="${OUT_DIR}/algorithm_analysis_${benchmark}_${mode}_${backend}.jsonl"

            if [ -f "$output_file" ]; then
                echo "[Skip] mode=${mode}, backend=${backend}, benchmark=${benchmark} (output exists: $output_file)"
                continue
            fi

            echo "================================================="
            echo "[Run] mode=${mode}, backend=${backend}, benchmark=${benchmark}"
            echo "================================================="
            python "analyze_algorithms_from_code_multi_backend.py" \
                --mode $mode --backend $backend --benchmark $benchmark

            sleep 5
        done
    done
done
