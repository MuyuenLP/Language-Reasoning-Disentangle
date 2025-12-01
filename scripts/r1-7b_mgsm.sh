export CUDA_VISIBLE_DEVICES=0,1
export RAY_TEMP_DIR="~/ray_vllm_temp"


PROJECT_DIR=~/Code/Neurips25/Language-Reasoning-Disentangle

MODEL_NAME_OR_PATH=~/Downloads/DeepSeek-R1-Distill-Qwen-7B
VECTOT_PATH=$PROJECT_DIR/vector/r1-distill-qwen-7b/parallel_multilingual_space/vector.pt
PROMPT_KEY=input

OUTPUT_ROOT_DIR=$PROJECT_DIR/results/r1-distill-qwen-7b/mgsm

STRENGTH_list=(0.3)
STRENGTH_list2=(0 -0.1 -0.2 0.3)
STEERING_layers="[10,11,12,13,14,15,16,17,18,19]"
STEERING_layers2="[20,21,22,23,24,25,26,27]"

EVAL_PATH=./FreeEvalLM
cd $EVAL_PATH


for STRENGTH in ${STRENGTH_list[@]}
do
    for STRENGTH2 in ${STRENGTH_list2[@]}
    do
        OUTPUT_DIR=$OUTPUT_ROOT_DIR/strength_${STRENGTH}_${STRENGTH2}
        mkdir -p $OUTPUT_DIR
        echo "Running with steering strength: $STRENGTH, $STRENGTH2"
        python $PROJECT_DIR/mlrs/src/main_generate_steering_task.py \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --vector_path $VECTOT_PATH  \
            --steering_strength "[$STRENGTH,$STRENGTH2]" \
            --task mgsm \
            --sample -1 \
            --output_dir $OUTPUT_DIR \
            --prompt_key $PROMPT_KEY \
            --reasoning_key reasoning \
            --reasoning_parser deepseek_r1 \
            --temperature 0.6 \
            --top_p 0.95 \
            --max_tokens 16384 \
            --tensor_parallel_size 2 \
            --steering_method Lens \
            --steering_layers $STEERING_layers \
            --steering_layers2 $STEERING_layers2
    
    done
done

# python $PROJECT_DIR/mlrs/src/main_summary_results_wtih_fidelity.py $OUTPUT_ROOT_DIR
# python $PROJECT_DIR/mlrs/src/grid_search_analyzer.py $OUTPUT_ROOT_DIR