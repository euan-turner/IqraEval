import os
import torchaudio
import torch
from datasets import load_dataset
import argparse

def main(args):
    dataset_path = args.path
    dataset_split = args.split
    output_dir = args.output_dir
    dev_name = args.dev_name

    # Load the dataset in streaming mode
    dataset = load_dataset(dataset_path, split=dataset_split, streaming=True)

    # Directories for output
    audio_dir = os.path.join(output_dir, dataset_split, "wav")
    audio_dir_dev = os.path.join(output_dir, dev_name, "wav")
    transcript_dir = os.path.join(output_dir, dataset_split, "transcripts")
    transcript_dir_dev = os.path.join(output_dir, dev_name, "transcripts")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(audio_dir_dev, exist_ok=True)
    os.makedirs(transcript_dir_dev, exist_ok=True)

    for idx, data in enumerate(dataset):
        # Extract audio
        audio_array = torch.tensor(data["audio"]["array"])
        sample_rate = data["audio"]["sampling_rate"]
        Spk_ID = data["speaker"]
        Spk_ID = Spk_ID.replace(" ", "_")
        if Spk_ID != dev_name:
            audio_file_path = os.path.join(audio_dir, f"audio_{Spk_ID}_{idx}.wav")
            torchaudio.save(audio_file_path, audio_array.unsqueeze(0), sample_rate)
            transcript = data["phoneme_aug"]

            # remove "<sil>" from transcript using join
            transcript = " ".join(x for x in transcript.split() if x != "<sil>")
            transcript_file_path = os.path.join(transcript_dir, f"transcript_{Spk_ID}_{idx}.txt")

            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write(transcript)
        else:
            audio_file_path = os.path.join(audio_dir_dev, f"audio_{Spk_ID}_{idx}.wav")
            torchaudio.save(audio_file_path, audio_array.unsqueeze(0), sample_rate)
            transcript = data["phoneme_aug"]

            # remove "<sil>" from transcript using join
            transcript = " ".join(x for x in transcript.split() if x != "<sil>")
            transcript_file_path = os.path.join(transcript_dir_dev, f"transcript_{Spk_ID}_{idx}.txt")
            
            with open(transcript_file_path, "w", encoding="utf-8") as f:
                f.write(transcript)

        # Print progress
        idx = int(idx)
        if idx % 100 == 0:
            print(f"Processed {idx} samples...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio and transcript data.")
    parser.add_argument(
        "--path",
        type=str,
        default="Yasel/SWS_TTS",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (e.g., dev, test, train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save outputs (audio and transcripts)",
    )
    parser.add_argument(
        "--dev_name",
        type=str,
        default="Amer",
        help="Speaker name to use as dev split (e.g., Amer)",
    )

    args = parser.parse_args()
    main(args)

# Example usage:
# python download_sdaia.py --path "mostafaashahin/SWS_TTS_v0.2" --split "train" --output_dir "./data" --dev_name "Amer"