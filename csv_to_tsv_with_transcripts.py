import os
import pandas as pd
import argparse

def main(args):
    csv_path = args.csv_path
    transcript_root = args.transcript_root
    output_path = args.output_path

    # Read CSV, assuming index_col=0 to skip 'Unnamed: 0'
    df = pd.read_csv(csv_path, index_col=0)
    print(df.head())

    transcripts = []
    for i in range(len(df)):
        # Replace 'wav/audio_' with 'transcripts/transcript_' and '.wav' with '.txt'
        text_file_name = df['file_path'][i].replace('wav/audio_', 'transcripts/transcript_').replace('.wav', '.txt')
        text_file_name = os.path.join(transcript_root, text_file_name)
        with open(text_file_name, 'r', encoding='utf-8') as file:
            transcript = file.read().replace('\n', '')
            transcripts.append(transcript)
        print(i)
    df['sentence'] = transcripts
    df = df[['file_path', 'sentence']]
    df = df.rename(columns={'file_path': 'path'})
    # Save to TSV
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved TSV with transcripts to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to TSV with inlined transcripts.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the input CSV file (with column 'file_path')."
    )
    parser.add_argument(
        "--transcript_root",
        type=str,
        required=True,
        help="Root directory to prepend to transcript text file paths."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output TSV file."
    )
    args = parser.parse_args()
    main(args)

# Example usage:
# python csv_to_tsv_with_transcripts.py --csv_path data/combined_sorted.csv --transcript_root /data_sdaia/ --output_path /data_sdaia/mixed/train.tsv