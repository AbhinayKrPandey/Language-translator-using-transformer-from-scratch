import torch
from torch.utils.data import Dataset
class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        src_tgt_pair = self.dataset[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        # Tokenize the source and target texts
        src_tokens = self.tokenizer_src.encode(src_text).ids
        tgt_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate padding (accounting for SOS and EOS tokens)
        enc_pad_len = self.seq_len - len(src_tokens) - 2  # [SOS] + ... + [EOS]
        dec_pad_len = self.seq_len - len(tgt_tokens) - 1  # [SOS] + ...

        if enc_pad_len < 0 or dec_pad_len < 0:
            raise ValueError(f"Input sequence too long (max {self.seq_len}). Source: {len(src_tokens)}, Target: {len(tgt_tokens)}")

        # Construct encoder input: [SOS] + tokens + [EOS] + [PAD] * n
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_pad_len, dtype=torch.int64)
        ],
        dim=0
        )

        # Construct decoder input: [SOS] + tokens + [PAD] * n
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64)
        ],
        dim=0
        )

        # Construct label: tokens + [EOS] + [PAD] * n
        label = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_pad_len, dtype=torch.int64)
        ],
        dim=0
        )

        # Sanity checks
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
