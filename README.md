# Interspeech-2025: 
# Step1: Data Download/ Prepare
## Install pip install datasets library:
```pip install pip install datasets ```
### Download CV-Ar data from huggingface and store into 
 - dev
   - transcripts
   - wav
 - train
   - transcripts
   - wav
 ```
python download_hugg_data.py --path "mostafaashahin/common_voice_Arabic_12.0_Augmented_SWS_lam_phoneme" --split "train" --output_dir "./sws_data/CV-Ar"
python download_hugg_data.py --path "mostafaashahin/common_voice_Arabic_12.0_Augmented_SWS_lam_phoneme" --split "dev" --output_dir "./sws_data/CV-Ar"

```
### Download Arabic TTS data described in the paper from huggingface and store into 
 - dev
   - transcripts
   - wav
 - train
   - transcripts
   - wav
```
python download_hugg_data_tts.py --path "mostafaashahin/SWS_TTS_v0.2" --split "train" --output_dir "./data/TTS" --dev_name "Amer"
```
## Install s3prl toolkit
```
conda create -n s3prl python=3.8 \
conda activate s3prl \
git clone https://github.com/s3prl/s3prl.git \
cd s3prl \
pip install -e ".[all]"
```
In this work, the downstream task located in the [s3prl/s3prl/downstream/ctc](https://github.com/s3prl/s3prl/tree/main/s3prl/downstream/ctc) is used to train a ctc based phoneme recognition system.

# Step2: Data setup for [s3prl/s3prl/downstream/ctc](https://github.com/s3prl/s3prl/tree/main/s3prl/downstream/ctc)

a.) 
Navigate to [s3prl/s3prl/preprocess](https://github.com/s3prl/s3prl/tree/main/s3prl/preprocess) and move  ``` generate_len_for_bucket_sdaia.py ``` there and run:

``` python generate_len_for_bucket_sdaia.py -i path_to_downloaded_data/sws_data/CV-Ar -o ../data/CV-Ar/```

This will generate sorted files inside ```s3prl/s3prl/data/CV-Ar/len_for_bucket/train.csv``` and ```s3prl/s3prl/data/CV-Ar/len_for_bucket/dev.csv```

b.) 
Navigate to [s3prl/s3prl](https://github.com/s3prl/s3prl/tree/main/s3prl/) and move  ``` csv_to_tsv_with_transcripts.py ``` there and run:

```
python csv_to_tsv_with_transcripts.py --csv_path s3prl/s3prl/data/CV-Ar/len_for_bucket/train.csv --transcript_root "./sws_data/CV-Ar/" --output_path "./sws_data/CV-Ar/train/train.tsv"
python csv_to_tsv_with_transcripts.py --csv_path s3prl/s3prl/data/CV-Ar/len_for_bucket/dev.csv --transcript_root "./sws_data/CV-Ar/" --output_path "./sws_data/CV-Ar/dev/dev.tsv"
```

# Step3: Setting up Downstream task: s3prl/s3prl/downstream/ctc

a). We need to create the vocab used for phoneme recognition training. The vocab for this dataset is available at: ```vocab/sws_arabic.txt``` or you can run the following code on training data to get the vocab:
```
python get_units.py --input_dir "path_to_your_data/transcripts" --output_path "path/units.txt"
```
b). Move ```vocab/sws_arabic.txt``` to ```s3prl/s3prl/downstream/ctc/cv_vocab/sws_arabic.txt```

c). Move the config/sws.yaml to  ```s3prl/s3prl/downstream/ctc/cv_config/sws.yaml```
The notable changes to the config are as follows:
 1. text.vocab_file
 2. downstream_expert.corpus.path: make sure to add your path to dataset
 3. downstream_expert.corpus.train,downstream_expert.corpus.dev, downstream_expert.corpus.test: make sure you add your tsv files for each split of the data
 4. text.mode: 'word', here we treat each phone similar to a separate word or token for ctc, this would also make wer error rate (wer) = phoneme error rate (per)

# Step4: Training and Evaluation bash script
```
exp_dir='hubert_base_per'
## training
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p ${exp_dir} -u hubert_base -d ctc
## evalaute
python3 run_downstream.py -m evaluate -e ${exp_dir}/dev-best.ckpt
```
This will give you ther PER for our task

# Step5: Running inference on our pre-trained checkpoints:
We provide the pre-trained checkpoints with Trainng data as "CV-Ar trainset + TTS " and validation set as "CV-Ar dev" set.
For HuBERT:
```
python s3prl_inference.py \
  --ckpt "https://huggingface.co/Trikaldarshi/sws_pretrained_models/resolve/main/hubert_base.ckpt" \
  --dict_path "https://huggingface.co/Trikaldarshi/sws_pretrained_models/resolve/main/sws_arabic.txt" \
  --wav_dir ./sws_data/CV-Ar/dev/wav/ \
  --output_csv results.csv
```
Please change:
 1. WavLM: wavlm_base.ckpt
 2. Wav2vec2: wav2vec2_base.ckpt
 3. mHuBERT: mhubert.ckpt

# Phoneme Alignment and Evaluation

This part provides utilities for aligning predicted phoneme sequences with ground truth and computing metrics (insertions, deletions, substitutions, correct rate, accuracy). 
---

## 📂 Folder Structure

```text
metric.py
align_data.py
ins_del_cor_sub_analysis.py
aligned/                  # Output directory for alignment details
results/                  # Directory for final summary reports
```

## 🔧 Step 1: Align Data

Use `align_data.py` to align predictions with both the canonical (reference) and annotated (human-transcribed) phoneme sequences.

```bash
python align_data.py \
  --truth-file /path/to/test.csv \
  --pred-file  /path/to/predictions.csv
```

This will create the `aligned/` folder (if it doesn’t exist) and write:

- `aligned/ref_our_detail` – Reference vs. Prediction  
- `aligned/ref_human_detail` – Reference vs. Human Annotation  
- `aligned/human_our_detail` – Human Annotation vs. Prediction  


## 📊 Step 2: Compute Metrics

Once the alignments are saved, run `ins_del_cor_sub_analysis.py` to compute summary metrics.

```bash
python ins_del_cor_sub_analysis.py \
  --out-file      results/out.txt \
  --align-folder  aligned/
```

- Reads alignment files from `aligned/`
- Computes Correct Rate & Accuracy
- Writes summary to `results/out.txt`

## 📝 Input CSV Formats

### `truth-file`
Must contain columns:
- `ID`
- `Reference_phn`
- `Annotated_phn`
- _Optional for vowel-only_: `Reference_vow_phn`, `Annotated_vow_phn`

### `pred-file`
Must contain columns:
- `ID`
- `Prediction`
