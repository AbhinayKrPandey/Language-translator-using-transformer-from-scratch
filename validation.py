import torch
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import warnings

# These files from the original project must be in the same directory
from config import get_config, get_weights_path
from model import build_transformer
from dataset import BilingualDataset, causal_mask
# This assumes run_validation is in train.py
from train import run_validation 

def get_dataset(config):
    # Load the training split as a validation set
    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    # Load tokenizers from the downloaded files
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    # Filter out long sentences for efficiency
    filtered_data = [
        item for item in dataset_raw
        if len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids) <= config['seq_len']
        and len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) <= config['seq_len']
    ]
    
    # Split the filtered data into a small validation set
    validation_dataset_size = int(len(filtered_data) * 0.1)  # Using 10% of the data
    val_data = filtered_data[:validation_dataset_size]

    val_dataset = BilingualDataset(val_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    return val_dataloader, tokenizer_src, tokenizer_tgt


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    # Load the validation dataset and tokenizers
    val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)

    # Build the model architecture
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        config['d_model']
    ).to(device)

    # Load your downloaded model weights
    # Make sure tmodel_01.pt is in a 'weights/' subfolder
    model_filename = "tmodel_01.pt" 
    state = torch.load(f"weights/{model_filename}", map_location=device)
    model.load_state_dict(state['model_state_dict'])

    print(f"Running validation for model: {model_filename}")

    # Run the validation process to get the BLEU score
    run_validation(
        model,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
        config['seq_len'],
        device,
        lambda msg: print(msg),
        0,
        None
    )