import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_weights_path, get_config
from tqdm import tqdm
import warnings
import torchmetrics


def greedy_decode(model,source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    #precompute the encoder output
    encoder_output = model.encode(source, source_mask)
    # initialize the decoder input with SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        #build decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate the decoder output
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get the next token
        prob = model.project(decoder_output[:, -1])
        _, next_word = torch.max(prob, dim = 1)

        # append the next token to the decoder input
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2 ):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    #Size of control window
    console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size should be 1 for validation"
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(src_text)
            expected.append(tgt_text)
            predicted.append(model_output_text)

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{src_text}")
            print_msg(f"{f'TARGET: ':>12}{tgt_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_output_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    # Moved metric calculations outside of the if writer: block
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)

    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)

    metric = torchmetrics.BLEUScore()
    bleu = metric([s.lower() for s in predicted], [[s.lower()] for s in expected])

    if writer:
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

        # ADDED: This line prints the BLEU score to the console
    print_msg(f"BLEU: {bleu.item():.2f}")



def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):

    dataset_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    #build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    filtered_data = [
        item for item in dataset_raw
        if len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids) <= config['seq_len']
        and len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) <= config['seq_len']
    ]

    #keep 90% of the data for training and 10% for validation
    train_dataset_size = int(len(filtered_data) * 0.9)
    validation_dataset_size = len(filtered_data) - train_dataset_size
    train_dataset_raw, validation_dataset_raw = random_split(filtered_data, [train_dataset_size, validation_dataset_size])

    train_data = [filtered_data[i] for i in train_dataset_raw.indices]
    val_data = [filtered_data[i] for i in validation_dataset_raw.indices]

    train_dataset = BilingualDataset(train_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    validation_dataset = BilingualDataset(val_data, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max source length: {max_len_src}, Max target length: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)  
    validation_dataloader = DataLoader(validation_dataset, batch_size= 1, shuffle= True)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        config['d_model'],
    )
    return model

def train_model(config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model.to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename =  get_weights_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (batch seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (batch, 1,1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (batch, 1, seq_len, seq_len)

            # run the tensor through the model
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)  # (batch, seq_len, vocab_tgt_len)

            label = batch['label'].to(device)  # (batch, seq_len)
            #(batch, seq_len, vocab_tgt_len)  --> (batch * seq_len, vocab_tgt_len)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            #log the loss to tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagation
            loss.backward()
            # update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none= True)

            global_step += 1
        
        run_validation(
                model,
                validation_dataloader,
                tokenizer_src,
                tokenizer_tgt,
                config['seq_len'],
                device,
                lambda msg: batch_iterator.write(msg),
                global_step,
                writer
            )

        # save the model
        model_filename = get_weights_path(config, f'{epoch:02d}')
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
