import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from transfromer_learn.models.util.utils import configs
from transfromer_learn.models.model.transformer import Transformer
from transfromer_learn.models.train.train import read_data
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

smoothingFunction = SmoothingFunction()

def load_model_tokenizer():
    '''
    hàm để load model và load tokenizer
    '''
    device = torch.device(configs["decvice"])
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"]) 
    
    # load model transformer
    model = Transformer(
        src_pad_idx=source_tokenizer.pad_token_id,
        trg_sos_idx=target_tokenizer.pad_token_id,
        enc_voc_size=source_tokenizer.vocab_size,
        dec_voc_size=target_tokenizer.vocab_size,
        embedding=configs["embedding_dim"],
        n_head=configs["n_heads"],
        max_length=configs["max_seq_len"],
        ffn_hidden=configs["hidden_dim"],
        n_layers=configs["n_layers"],
        dropout=configs["dropout"],
        device=configs["device"]
    )
    
    model.load_state_dict(torch.load(configs["model_path"]))
    model.eval()
    model.to(device=device)
    print(f"Done load model on the {device} device")  
    return model, source_tokenizer, target_tokenizer



def translate(model, sentence, source_tokenizer, target_tokenizer, device=torch.device("cpu") print_process=True):
    """
    hàm chuyển seq nguồn thành token và sinh seq từ model
    """
    source_tokens = source_tokenizer.encode(sentence)[:configs["source_max_seq_len"]]
    
    # thêm chiều batch 
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device=device)
    