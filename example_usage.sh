#!/bin/bash

# Example usage of run_iqra.py script
# This script demonstrates how to run the evaluation
# Note: Dictionary path is hardcoded to vocab/sws_arabic.txt

# Example 1: Using pre-trained models in test mode (inference only)
echo "Example 1: Using pre-trained HuBERT model in test mode"
python run_iqra.py \
    --model hubert_base \
    --mode test \
    --max_samples 10

echo "Example 2: Using pre-trained mHuBERT model in eval mode"
python run_iqra.py \
    --model mhubert_base \
    --mode eval \
    --max_samples 10

echo "Example 3: Using pre-trained Wav2Vec2 model in eval mode"
python run_iqra.py \
    --model wav2vec2_base \
    --mode eval \
    --max_samples 10

echo "Example 4: Using pre-trained WavLM model in eval mode"
python run_iqra.py \
    --model wavlm_base \
    --mode eval \
    --max_samples 10

# Example 5: Full evaluation on dev set
echo "Example 5: Full evaluation on dev set"
python run_iqra.py \
    --model hubert_base \
    --mode eval

# Example 6: Using custom model checkpoint
echo "Example 6: Using custom model checkpoint"
python run_iqra.py \
    --ckpt /path/to/custom/model.ckpt \
    --mode eval \
    --max_samples 10

# Example 7: Using custom model with custom dictionary
echo "Example 7: Custom model with custom dictionary"
python run_iqra.py \
    --ckpt /path/to/custom/model.ckpt \
    --dict_path /path/to/custom/dict.txt \
    --mode eval \
    --max_samples 10

# Example 8: Test mode for quick inference check
echo "Example 8: Test mode for quick inference check"
python run_iqra.py \
    --model hubert_base \
    --mode test \
    --max_samples 5
