import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re


class TranslateDataset(Dataset):
    
    def __init__(self, source_tokenizer, target_tokenizer, source_data=None, target_data=None, source_max_length=256, target_max_length=256, phase="train"):
        self.source_data = source_data
        self.target_data = target_data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.phase = phase
        
    
    def preprocess_seq(self, seq):
        '''
        xu ly du lieu dau vao dang chuoi
        '''
        seq = re.sub(
             r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq)
        )
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r",+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        return seq.strip().lower()
    
    
    def conert_line_uncase(self, tokenizer, text, max_seq_len):
        '''
        chuyen van ban thanh token
        '''
        tokens = tokenizer.tokenize(text)[:max_seq_len - 2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens += [tokenizer.pad_token] * (max_seq_len - len(tokens))
        token_idx = tokenizer.convert_tokens_to_idx(tokens)
        return tokens, token_idx
    
    def __len__(self):
        return len(self.source_data)
    
    def __getitem__(self, index):
    
        source = self.source_tokenizer(
            text = self.preprocess_seq(self.source_data[index]),
            padding = "max_length",
            max_length = self.source_max_length,
            truncation = True,
            return_tensors = "pt"
        )
    
        if self.phase == "train":
            target = self.target_tokenizer(
                text=self.preprocess_seq(self.target_data[index]),
                padding="max_length",
                max_length=self.target_max_seq_len,
                truncation=True,
                return_tensors="pt"
            )

            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
                "target_seq": self.target_data[index],
                "target_ids": target["input_ids"][0],
                }
        else:
            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
            }