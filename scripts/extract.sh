export CUDA_VISIBLE_DEVICES=0

SRC_DIR=.
MODEL_NAME_OR_PATH=/data/Downloads/DeepSeek-R1-Distill-Qwen-7B


INPUT_PATH=./data/parallel_multilingual_math
langs=$(find "$INPUT_PATH" -type f -name "*.json" | xargs -n1 basename | sed 's/\.json$//')
echo $langs
# 遍历这个列表
for lang in $langs; do
    echo "$lang"
    echo ${INPUT_PATH}/${lang}.json
    OUTPUT_DIR=./vector/r1-distill-qwen-7b/parallel_multilingual_math/$lang
    mkdir -p $OUTPUT_DIR
    python $SRC_DIR/mlrs/src/main_extract_activations.py \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --input_path ${INPUT_PATH}/${lang}.json \
            --output_dir ${OUTPUT_DIR} \
            --prompt_key translated_prompt \
            --reasoning_key thoughts \
            --reasoning_parser deepseek_r1 \
            --temperature 0.6 \
            --top_p 0.95 \
            --if_think False
done

python $SRC_DIR/mlrs/src/main_multilingual_space.py \
    --acts_path ./vector/r1-distill-qwen-7b/parallel_multilingual_math \
    --output_path ./vector/r1-distill-qwen-7b/parallel_multilingual_space/vector.pt