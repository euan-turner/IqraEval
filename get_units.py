import os
import re
import argparse

def main(args):
    input_dir = args.input_dir
    output_path = args.output_path

    # get the list of all the .txt files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # create a set to store the unique units/words
    units = set()

    # read each file and get the unique units/words
    for file in files:
        with open(os.path.join(input_dir, file), 'r', encoding="utf-8") as f:
            for line in f:
                # split the line into words by any whitespace
                words = re.split(r'\s+', line.strip())
                for word in words:
                    if word:  # avoid empty strings
                        units.add(word)

    # print stats
    print(f"Total unique units: {len(units)}")
    # Optionally print the units
    # print(units)
    # for unit in units:
    #     print(unit)

    # Save all units in a file, one per line
    with open(output_path, "w", encoding="utf-8") as out_f:
        for unit in sorted(units):
            out_f.write(unit + "\n")
    print(f"Units saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unique units/words from text files in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing transcript .txt files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the units.txt file",
    )
    args = parser.parse_args()
    main(args)

# Example usage:
# python get_units.py --input_dir "path_to_your_data/transcripts" --output_path "path/units.txt"
