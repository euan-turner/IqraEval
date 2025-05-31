import os
import torchaudio
import torch
from datasets import load_dataset
import argparse

def main(args):
    dataset_path = args.path
    dataset_split = args.split
    output_dir = args.output_dir

    dataset = load_dataset(dataset_path, split=dataset_split, streaming=True)

    audio_dir = os.path.join(output_dir, dataset_split, "wav")
    transcript_dir = os.path.join(output_dir, dataset_split, "transcripts")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    for idx, data in enumerate(dataset):
        audio_array = torch.tensor(data["audio"]["array"])
        sample_rate = data["audio"]["sampling_rate"]
        ID = data["ID"]
        audio_file_path = os.path.join(audio_dir, f"audio_{ID}.wav")
        torchaudio.save(audio_file_path, audio_array.unsqueeze(0), sample_rate)

        transcript = data["phoneme"]
        transcript_file_path = os.path.join(transcript_dir, f"transcript_{ID}.txt")
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        if idx % 100 == 0:
            print(f"Processed {idx} samples...")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio and transcript data.")
    parser.add_argument(
        "--path",
        type=str,
        default="mostafaashahin/common_voice_Arabic_12.0_Augmented_SWS_lam_phoneme",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to use (e.g., dev, test, train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save outputs (audio and transcripts)",
    )

    args = parser.parse_args()
    main(args)

# # Print more usage examples
# print('python download_sdaia.py --path "mostafaashahin/common_voice_Arabic_12.0_Augmented_SWS_lam_phoneme" --split "train" --output_dir "./data"')
# print('python download_sdaia.py --path "mostafaashahin/common_voice_Arabic_12.0_Augmented_SWS_lam_phoneme" --split "dev" --output_dir "./data"')
