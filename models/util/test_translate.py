import torch
import time
from transfromer_learn.models.util.utils import configs
from transfromer_learn.models.util.evaluate import load_model_tokenizer, translate


# test với 1 câu bất kì để model dịch
def main():
    # translate sentence
    sentence = "My duty was not to allow it to have been in vain , and my lesson was to learn that , yes , history tried to crush us , but we endured ."
    print("--- English input sentence:", sentence)
    print("--- Translating...")
    device = torch.device(configs["device"])
    model, source_tokenizer, target_tokenizer = load_model_tokenizer(configs)
    st = time.time()
    trans_sen = translate(
        model=model, 
        sentence=sentence, 
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"], 
        beam_size=configs["beam_size"], 
        device=device
    )
    end = time.time()
    print("--- Sentences translated into Vietnamese:", trans_sen)
    print(f"--- Time: {end-st} (s)")
    
    
if __name__ == "__main__":
    main()