# IqraEval - Evaluation Script

This repository contains evaluation scripts for S3PRL models on the IqraEval dataset.

## Files

- `run_iqra.py`: **Main evaluation script** with test and eval modes
- `s3prl_inference.py`: Original S3PRL inference script
- `example_usage.sh`: Example commands for running the evaluation

## Data Download

### Install datasets library

```bash
pip install datasets
```

### Download CV-Ar data from huggingface and store into

- dev
  - transcripts
  - wav
- train
  - transcripts
  - wav

```bash
python download_hugg_data.py --path "IqraEval/Iqra_train" --split "train" --output_dir "./sws_data/CV-Ar"
python download_hugg_data.py --path "IqraEval/Iqra_train" --split "dev" --output_dir "./sws_data/CV-Ar"
```

### Download Arabic TTS data described in the paper from huggingface and store into

- dev
  - transcripts
  - wav
- train
  - transcripts
  - wav

```bash
python download_hugg_data_tts.py --path "IqraEval/Iqra_TTS" --split "train" --output_dir "./data/TTS" --dev_name "Amer"
```

## Features

The `run_iqra.py` script provides a simplified, focused evaluation experience:

### Two Modes of Operation

1. **Test Mode** (default): Inference only, displays predictions without comparison
2. **Eval Mode**: Full evaluation with ground truth comparison and error statistics

### Key Features

- **Simple Model Loading**: Streamlined S3PRL model loading
- **Automatic Dataset Handling**: Loads `IqraEval/Iqra_train` dev set automatically
- **Audio Format Support**: Handles torchcodec AudioDecoder format from HuggingFace datasets
- **Error Statistics**: Comprehensive phoneme error analysis in eval mode
- **Clean Output**: Clear, readable results display

## Usage

### Basic Usage

```bash
# Test mode - inference only (default)
python run_iqra.py --max_samples 10

# Eval mode - with ground truth comparison and error statistics  
python run_iqra.py --mode eval --max_samples 10

# Full evaluation on entire dev set
python run_iqra.py --mode eval
```

### Model Selection

```bash
# Use different pre-trained models
python run_iqra.py --model hubert_base --mode eval
python run_iqra.py --model mhubert_base --mode eval  
python run_iqra.py --model wav2vec2_base --mode eval
python run_iqra.py --model wavlm_base --mode eval

# Use custom checkpoint
python run_iqra.py --ckpt path/to/model.ckpt --mode eval
```

### Arguments (run_iqra.py)

- `--model`: Pre-trained model to use (hubert_base, mhubert_base, wav2vec2_base, wavlm_base)
- `--ckpt`: Path or URL to custom S3PRL model checkpoint (alternative to --model)
- `--dict_path`: Path to dictionary file (default: vocab/sws_arabic.txt)
- `--mode`: Operation mode - "test" for inference only, "eval" for evaluation with ground truth
- `--max_samples`: Maximum number of samples to evaluate (optional, for testing)

### Output Formats

#### Test Mode Output

- Displays predictions for first 5 samples
- Shows sentence, tashkeel, ground truth, and predictions
- Summary statistics (average phoneme lengths)

#### Eval Mode Output

- Same as test mode plus:
- Loss values for each sample
- Phoneme Error Rate (PER) per sample and overall
- Detailed error statistics (substitutions, insertions, deletions)
- Overall evaluation metrics

### Example Output

```text
--- Sample 1 ---
Index: 0
Sentence: الآن قد استوعبتها
Tashkeel: الْآنَ قَدِ اسْتَوْعَبْتُهَا
Ground Truth Phonemes: < a l < aa n a q A d i s t a w E a b t u h aa
Predicted Phonemes: < a l < aa n a q A s t a w E a b t h aa
Loss: 176.6103
PER: 0.1364
Correct: 19/22
Errors - Sub: 0, Ins: 3, Del: 0
```

## Dependencies

- torch
- torchaudio
- s3prl
- datasets
- pandas
- tqdm
- requests
- tensorboardX
- editdistance
