<div align="center">
  <h1>When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners</h1>
  <a href="https://neurips.cc/">
    <img src="https://img.shields.io/badge/Conference-NeurIPS'25-green" alt="NeurIPS 2025 Conference"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15257">
    <img src="https://img.shields.io/badge/Status-Accepted-success" alt="Paper Status Accepted"/>
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11.5-blue.svg" alt="Python Version"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.5.1-red.svg" alt="PyTorch Version"/>
  </a>

  <strong>The official implementation for the NeurIPS 2025 💡 Spotlight Paper</strong>  
  <em>"When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners"</em>

  <p>
    <a href="https://arxiv.org/abs/2505.15257">📄 Paper</a> • 
    <a href="https://github.com/MuyuenLP/Language-Reasoning-Disentangle">🚀 Code</a>
  </p>
</div>


---


## 🔧 Requirements

* Python 3.11.5
* PyTorch 2.5.1
* Transformers 4.51.3
* vllm 0.7.3

For other missing packages, simply use pip to install them.

### Installation

```bash
git clone https://github.com/MuyuenLP/Language-Reasoning-Disentangle.git
cd Language-Reasoning-Disentangle
pip install -e .
cd ./FreeEvalLM
pip install -e .
```

## 🚀 Quick Start

### 1. Extract Activations
Extract hidden state activations from multilingual data using the same prompts across different languages:

```bash
# Extract activations for each language
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
```

### 2. Build Language Space
Extract language-specific vectors using SVD decomposition from multilingual activations:

```bash
python $SRC_DIR/mlrs/src/main_multilingual_space.py \
    --acts_path ./vector/r1-distill-qwen-7b/parallel_multilingual_math \
    --output_path ./vector/r1-distill-qwen-7b/parallel_multilingual_space/vector.pt
```

### 3. Disentangle and Re-inject
Perform language-reasoning disentanglement during inference:

```bash
python mlrs/src/main_generate_steering_task.py \
    --model_name_or_path /path/to/your/model \          # Path to the target model
    --vector_path ./vectors/your_model/vector.pt \       # Language-specific vectors from step 2
    --steering_strength "[0.3, -0.1]" \                 # Intervention strengths [layer_group1, layer_group2]
    --task mgsm \                                        # Evaluation task (mgsm/xwinograd_for_mlrs/mmmlu_for_mlrs)
    --output_dir ./results/your_model/mgsm \             # Output directory for results
    --steering_method Mlrs \                             # Disentanglement method (Mlrs/vanilla)
    --steering_layers "[10,11,12,13,14,15,16,17,18,19]" \    # Lower-middle layers for reasoning enhancement
    --steering_layers2 "[20,21,22,23,24,25,26,27]"          # Upper layers for language fidelity preservation
```

**Parameter Explanations:**
- `steering_strength`: Two values for different layer groups. Positive values represent Disentangle, negative values represent Reinject.
- `steering_layers`: Middle layers where reasoning enhancement is applied
- `steering_layers2`: Upper layers where language-specific signals are lightly adjusted to maintain fidelity
- `steering_method`: "Mlrs" uses projection-based disentanglement, "vanilla" uses simple vector addition

You can also directly run the full script at `scripts/r1-7b_mgsm.sh`



### 4. Evaluate Results
Summarize results and analyze performance:

```bash
# Summarize results with language fidelity
python mlrs/src/main_summary_results_wtih_fidelity.py ./results/your_model/mgsm
# Analyze grid search results
python mlrs/src/grid_search_analyzer.py ./results/your_model/mgsm
```


## 📚 Citation

If you find our work useful for your research, please kindly cite our paper:

```bibtex
Coming Soon
```
