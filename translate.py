from pathlib import Path
from config import get_config, get_weights_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
from train import greedy_decode
import torch
import sys



def translate(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    model_filename = get_weights_path(config, f'31')
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='all')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    model.eval()
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source = source.unsqueeze(0)  # Add batch dimension
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).int().to(device)
        output_ids = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
    return tokenizer_tgt.decode(output_ids.tolist())
    
#read sentence from argument
translate(sys.argv[1] if len(sys.argv) > 1 else "I am Abhinay")