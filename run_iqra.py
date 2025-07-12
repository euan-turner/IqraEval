import torch
import argparse
import tempfile
import requests
import os
from datasets import load_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def download_if_needed(path):
    """Download file if path is a URL, else return path unchanged."""
    if str(path).startswith("http://") or str(path).startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        suffix = os.path.splitext(str(path))[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(response.content)
        tmp.close()
        return tmp.name
    return path

def load_model(ckpt_path, dict_path):
    """Load model from checkpoint and dictionary files."""
    from s3prl.downstream.runner import Runner
    
    # Download files if needed
    ckpt = download_if_needed(ckpt_path)
    dict_path = download_if_needed(dict_path)
    
    # Load model checkpoint
    torch.serialization.add_safe_globals([argparse.Namespace])
    model_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
    args = model_dict['Args']
    config = model_dict['Config']
    
    # Config patch
    args.init_ckpt = ckpt
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    config['downstream_expert']['text']["vocab_file"] = dict_path
    config['runner']['upstream_finetune'] = False
    config['runner']['layer_drop'] = False
    config['runner']['downstream_pretrained'] = None
    
    # Initialize model components
    runner = Runner(args, config)
    upstream = runner._get_upstream()
    featurizer = runner._get_featurizer()
    downstream = runner._get_downstream()
    
    # Track temp files for cleanup
    temp_files = []
    if ckpt.startswith(tempfile.gettempdir()):
        temp_files.append(ckpt)
    if dict_path.startswith(tempfile.gettempdir()):
        temp_files.append(dict_path)
    
    model_components = {
        'upstream': upstream,
        'featurizer': featurizer,
        'downstream': downstream,
        'args': args,
        'device': args.device
    }
    
    print(f"Model loaded on device: {args.device}")
    return model_components, temp_files

def predict_phonemes(model_components, audio_array, sample_rate, mode="test", ground_truth=None):
    """
    Predict phonemes from audio array.
    
    Args:
        model_components: Dictionary containing model components
        audio_array: Audio waveform as numpy array
        sample_rate: Sample rate of audio
        mode: "test" for inference only, "eval" for evaluation with ground truth
        ground_truth: Ground truth phonemes (required for eval mode)
        
    Returns:
        Dict with predictions and optionally loss/metrics
    """
    try:
        # Convert to tensor and move to device
        # Handle different audio array types
        if hasattr(audio_array, 'numpy'):
            # If it's a tensor, convert to numpy first
            audio_array = audio_array.numpy()
        
        # Ensure it's a numpy array
        import numpy as np
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)
        
        # Ensure it's the right shape (flatten if needed)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.flatten()
        
        wav = torch.FloatTensor(audio_array).unsqueeze(0)
        if len(wav.shape) > 2:  # If stereo, convert to mono
            wav = wav.mean(1).unsqueeze(0)
        wav = wav.to(model_components['device'])
        
        # Prepare inputs based on mode
        if mode == "eval" and ground_truth:
            # For eval mode, still use inference but we'll calculate metrics separately
            # The S3PRL downstream model may not handle eval mode properly during inference
            split = "inference"  # Use inference mode but calculate loss separately
            filenames = ["eval_sample"]
            records = {"loss": [], "hypothesis": [], "groundtruth": [], "filename": []}
        else:
            # Use dummy inputs for inference
            split = "inference"
            filenames = ["inference_sample"]
            records = {"loss": [], "hypothesis": [], "groundtruth": [], "filename": []}
        
        with torch.no_grad():
            features = model_components['upstream'].model(wav)
            features = model_components['featurizer'].model(wav, features)
            
            # Always use empty labels for inference (we calculate metrics separately for eval mode)
            dummy_labels = [[] for _ in features]
                
            model_components['downstream'].model(split, features, dummy_labels, filenames, records)
            predictions = records["hypothesis"]
            loss = records["loss"]
        
        result = {
            'predictions': predictions[0] if predictions else [],
            'loss': loss[0] if loss else None
        }
        
        return result
        
    except Exception as e:
        print(f"Error in predict_phonemes: {str(e)}")
        print(f"Audio array type: {type(audio_array)}")
        print(f"Audio array shape: {audio_array.shape if hasattr(audio_array, 'shape') else 'No shape'}")
        print(f"Ground truth type: {type(ground_truth)}")
        raise

def calculate_error_statistics(predictions, ground_truth):
    """
    Calculate phoneme error statistics.
    
    Args:
        predictions: List of predicted phonemes
        ground_truth: List of ground truth phonemes
        
    Returns:
        Dict with error statistics
    """
    if not predictions or not ground_truth:
        return {'per': 1.0, 'correct': 0, 'total': len(ground_truth), 'substitutions': 0, 'insertions': 0, 'deletions': 0}
    
    # Simple alignment using dynamic programming (edit distance)
    pred_phonemes = predictions if isinstance(predictions, list) else predictions.split()
    gt_phonemes = ground_truth if isinstance(ground_truth, list) else ground_truth.split()
    
    # Calculate edit distance and operations
    m, n = len(pred_phonemes), len(gt_phonemes)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_phonemes[i-1] == gt_phonemes[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # deletion
                    dp[i][j-1],     # insertion
                    dp[i-1][j-1]    # substitution
                )
    
    # Backtrack to count operations
    i, j = m, n
    substitutions = insertions = deletions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred_phonemes[i-1] == gt_phonemes[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            insertions += 1
            j -= 1
    
    total_errors = substitutions + insertions + deletions
    correct = n - total_errors
    per = total_errors / n if n > 0 else 1.0
    
    return {
        'per': per,
        'correct': correct,
        'total': n,
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions
    }

def evaluate_model(model_components, dataset, mode="test", max_samples=None):
    """
    Evaluate model on dataset and display results.
    
    Args:
        model_components: Dictionary containing model components
        dataset: HuggingFace dataset
        mode: "test" for inference only, "eval" for evaluation with ground truth
        max_samples: Maximum number of samples to process (for testing)
    """
    print(f"Starting evaluation in {mode} mode...")
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited evaluation to {len(dataset)} samples")
    
    results = []
    total_loss = 0.0
    total_errors = {'per': 0.0, 'correct': 0, 'total': 0, 'substitutions': 0, 'insertions': 0, 'deletions': 0}
    
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Extract audio information - handle different audio data types
            audio_data = sample['audio']
            
            # Handle different audio data formats
            if hasattr(audio_data, 'get_all_samples'):
                # If it's an AudioDecoder from torchcodec, get all samples
                decoded_audio = audio_data.get_all_samples()
                audio_array = decoded_audio.data.numpy().squeeze()  # Convert to numpy and remove extra dims
                sample_rate = audio_data.metadata.sample_rate
            elif hasattr(audio_data, 'decode'):
                # If it's an AudioDecoder with decode method
                decoded_audio = audio_data.decode()
                audio_array = decoded_audio['array']
                sample_rate = decoded_audio['sampling_rate']
            elif isinstance(audio_data, dict):
                # If it's already a dict
                audio_array = audio_data['array']
                sample_rate = audio_data['sampling_rate']
            else:
                # Try to access as attributes
                audio_array = audio_data.array if hasattr(audio_data, 'array') else audio_data
                sample_rate = audio_data.sampling_rate if hasattr(audio_data, 'sampling_rate') else 16000
            
            # Get ground truth and other info
            ground_truth = sample.get('phoneme', '')
            sentence = sample.get('sentence', '')
            tashkeel_sentence = sample.get('tashkeel_sentence', '')
            index = sample.get('index', i)
            
            # Predict phonemes
            prediction_result = predict_phonemes(
                model_components, audio_array, sample_rate, 
                mode=mode, ground_truth=ground_truth if mode == "eval" else None
            )
            
            predictions = prediction_result['predictions']
            loss = prediction_result['loss']
            
            # Calculate error statistics for eval mode
            error_stats = None
            if mode == "eval":
                error_stats = calculate_error_statistics(predictions, ground_truth)
                # Accumulate statistics
                total_errors['per'] += error_stats['per']
                total_errors['correct'] += error_stats['correct']
                total_errors['total'] += error_stats['total']
                total_errors['substitutions'] += error_stats['substitutions']
                total_errors['insertions'] += error_stats['insertions']
                total_errors['deletions'] += error_stats['deletions']
                
                if loss is not None:
                    total_loss += loss
            
            # Store results
            result = {
                'index': index,
                'sentence': sentence,
                'tashkeel_sentence': tashkeel_sentence,
                'ground_truth_phonemes': ground_truth,
                'predicted_phonemes': ' '.join(predictions) if isinstance(predictions, list) else str(predictions),
                'loss': loss,
                'error_stats': error_stats
            }
            results.append(result)
            
            # Display results for first few samples
            if i < 5:
                print(f"\n--- Sample {i+1} ---")
                print(f"Index: {index}")
                print(f"Sentence: {sentence}")
                print(f"Tashkeel: {tashkeel_sentence}")
                print(f"Ground Truth Phonemes: {ground_truth}")
                print(f"Predicted Phonemes: {result['predicted_phonemes']}")
                
                if mode == "eval" and error_stats:
                    print(f"Loss: {loss:.4f}" if loss is not None else "Loss: N/A")
                    print(f"PER: {error_stats['per']:.4f}")
                    print(f"Correct: {error_stats['correct']}/{error_stats['total']}")
                    print(f"Errors - Sub: {error_stats['substitutions']}, Ins: {error_stats['insertions']}, Del: {error_stats['deletions']}")
                
                print("-" * 50)
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    print(f"\nEvaluation complete! Processed {len(results)} samples")
    
    # Display summary statistics
    if results:
        print("\nSummary:")
        print(f"Total samples processed: {len(results)}")
        print(f"Average predicted phoneme length: {sum(len(r['predicted_phonemes'].split()) for r in results) / len(results):.2f}")
        print(f"Average ground truth phoneme length: {sum(len(r['ground_truth_phonemes'].split()) for r in results) / len(results):.2f}")
        
        if mode == "eval":
            avg_per = total_errors['per'] / len(results)
            avg_loss = total_loss / len(results) if total_loss > 0 else 0
            
            print("\nEvaluation Statistics:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average PER: {avg_per:.4f}")
            print(f"Total Correct Phonemes: {total_errors['correct']}/{total_errors['total']}")
            print(f"Total Errors - Substitutions: {total_errors['substitutions']}, Insertions: {total_errors['insertions']}, Deletions: {total_errors['deletions']}")
            print(f"Overall PER: {(total_errors['substitutions'] + total_errors['insertions'] + total_errors['deletions']) / total_errors['total']:.4f}")
    
    return results

def cleanup_temp_files(temp_files):
    """Clean up downloaded temporary files."""
    for f in temp_files:
        if f and os.path.isfile(f):
            os.remove(f)
            print(f"Cleaned up temporary file: {f}")

def main():
    # Valid pre-trained model configurations
    DICT_PATH = "vocab/sws_arabic.txt"
    
    VALID_MODELS = {
        "hubert_base": "https://huggingface.co/IqraEval/Iqra_hubert_base/resolve/main/hubert_base.ckpt",
        "mhubert_base": "https://huggingface.co/IqraEval/Iqra_mhubert_base/resolve/main/mhubert.ckpt",
        "wav2vec2_base": "https://huggingface.co/IqraEval/Iqra_wav2vec2_base/resolve/main/wave2vec2_base.ckpt",
        "wavlm_base": "https://huggingface.co/IqraEval/Iqra_wavlm_base/resolve/main/wavlm_base.ckpt"
    }
    
    parser = argparse.ArgumentParser(description="Clean evaluation of S3PRL model on IqraEval dataset.")
    parser.add_argument("--model", type=str, choices=list(VALID_MODELS.keys()), 
                       default="hubert_base",
                       help="Pre-trained model to use. Choose from: " + ", ".join(VALID_MODELS.keys()))
    parser.add_argument("--ckpt", type=str, 
                       help="Path or URL to S3PRL model checkpoint (alternative to --model).")
    parser.add_argument("--dict_path", type=str, default=DICT_PATH,
                       help=f"Path to dictionary file (default: {DICT_PATH}).")
    parser.add_argument("--mode", type=str, choices=["test", "eval"], default="test",
                       help="Mode: 'test' for inference only, 'eval' for evaluation with ground truth (default: test).")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing).")
    
    args = parser.parse_args()
    
    # Determine model paths
    if args.ckpt:
        ckpt_path = args.ckpt
        dict_path = args.dict_path
        print(f"Using custom model: {ckpt_path}")
    else:
        ckpt_path = VALID_MODELS[args.model]
        dict_path = DICT_PATH
        print(f"Using pre-trained model: {args.model}")
    
    print(f"Dictionary path: {dict_path}")
    
    temp_files = []
    try:
        # 1. Load the S3PRL model
        print("Loading S3PRL model...")
        model_components, temp_files = load_model(ckpt_path, dict_path)
        
        # 2. Load the dataset
        print("Loading IqraEval dataset...")
        dataset = load_dataset("IqraEval/Iqra_train", split="dev")
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Display dataset info
        print("\nDataset columns:", dataset.column_names)
        print("Sample keys:", list(dataset[0].keys()))
        
        # 3. Perform evaluation
        evaluate_model(model_components, dataset, args.mode, args.max_samples)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)

if __name__ == "__main__":
    main()
