import torch
import torchaudio
import s3prl
import warnings
import pandas as pd
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

class S3PRLModel:
    def __init__(self, ckpt, dict_path='dict.txt'):
        from s3prl.downstream.runner import Runner
        model_dict = torch.load(ckpt, map_location='cpu')
        self.args = model_dict['Args']
        self.config = model_dict['Config']
        
        # Config patch
        self.args.init_ckpt = ckpt
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config['downstream_expert']['text']["vocab_file"] = dict_path
        # config.runner.upstream_finetune=False
        # config.runner.layer_drop=False,,
        # config.runner.downstream_pretrained=None,,
        self.config['runner']['upstream_finetune']=False
        self.config['runner']['layer_drop']=False
        self.config['runner']['downstream_pretrained']=None
        
        runner = Runner(self.args, self.config)
        self.upstream = runner._get_upstream()
        self.featurizer = runner._get_featurizer()
        self.downstream = runner._get_downstream()

    def __call__(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0).unsqueeze(0)  # Convert to mono
        wav = wav.to(self.args.device)
        
        # Prepare dummy inputs
        dummy_split = "inference"
        dummy_filenames = [Path(wav_path).stem]  # Use filename as ID
        dummy_records = {"loss": [], "hypothesis": [], "groundtruth": [], "filename": []}
        
        with torch.no_grad():
            features = self.upstream.model(wav)
            features = self.featurizer.model(wav, features)
            dummy_labels = [[] for _ in features]  # Empty labels
            self.downstream.model(dummy_split, features, dummy_labels, dummy_filenames, dummy_records)
            predictions = dummy_records["hypothesis"]
        
        return predictions

def process_directory(ckpt, dict_path, wav_dir, output_csv):
    model = S3PRLModel(ckpt, dict_path)
    wav_files = list(Path(wav_dir).glob("*.wav"))
    results = []
    
    for wav_file in wav_files:
        output = model(str(wav_file))
        print(output)
        results.append({"filename": wav_file.name, "output": output})
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WAV files using S3PRL model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to dictionary file.")
    parser.add_argument("--wav_dir", type=str, required=True, help="Directory containing WAV files.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file.")
    
    args = parser.parse_args()
    process_directory(args.ckpt, args.dict_path, args.wav_dir, args.output_csv)
