import torch
import torch.nn as nn
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
import os

from transfromer_learn.models.model.transformer import Transformer
from transfromer_learn.models.util.utils import configs, plot_loss
from transfromer_learn.models.util.loader_data import TranslateDataset

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# đọc file 
def read_data(source_file, target_file):
    source_data = open(source_file).read().strip().split("\n")
    target_data = open(target_file).read().strip().split("\n")
    return source_data, target_data


# hàm train từng epoch
def train_epoch(model, train_loader, optim, epoch, n_epochs, target_pad_id, device):
    model.train() # để pytroch biết bật chế độ training
    total_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        # lấy batch source, target 
        source = batch["source_ids"].to(device)
        target = batch["target_ids"].to(device)
        
        # input train đầu decoder
        target_input = target[:,:,-1]
        
        # out slice nhãn
        gold = target[:,1,:].contiguous().view(-1)
        
        optim.zero_grad()
        
        # giúp train nhanh hơn và giảm vram
        with autocast():
            preds = model(source, target_input)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
    
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer=optim)
        scaler.update()
        
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])
    
    return sum(total_loss) / len(total_loss), total_loss

# hàm valid
def valid_epoch(model, valid_loader, epoch, n_epochs, target_pad_id, device):
    model.eval() # để pytorch bật chế độ val
    total_loss = []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}/{n_epochs}")
    
    for i, batch in bar:
    # lấy batch source, target 
        source = batch["source_ids"].to(device)
        target = batch["target_ids"].to(device)
        
        # input train đầu decoder
        target_input = target[:,:,-1]
        
        # out slice nhãn
        gold = target[:,1,:].contiguous().view(-1)
        
        preds = model(source, target_input)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])
    
    return sum(total_loss) / len(total_loss), total_loss


def train(model, train_loader, valid_loader, optim, n_epochs, target_pad_id, device, model_path, early_stopping):
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    best_val_loss = np.inf
    best_epoch = 1
    count_early_stop = 0
    log = {"train_loss": [], "valid_loss": [], "train_batch_loss": [], "valid_batch_loss": []}
    for epoch in range(n_epochs):
        train_loss, train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optim=optim,
            epoch=epoch,
            n_epochs=n_epochs,
            target_pad_id=target_pad_id,
            device=device
        )
        
        valid_loss, valid_losses = valid_epoch(
            model=model,
            valid_loader=valid_loader,
            epoch=epoch,
            n_epochs=n_epochs,
            target_pad_id=target_pad_id
        )
        
        # điều kiện dừng training khi overfiting
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_epoch = epoch + 1
            # save model 
            torch.save(model.state_dict(), model_path)
            print("---- Detect improment and save the best model ----")
            count_early_stop = 0
        else:
            count_early_stop += 1
            if count_early_stop >= early_stopping:
                print("---- Early stopping ----")
                break
            
        torch.cuda.empty_cache()           
        log["train_loss"].append(train_loss)
        log["valid_loss"].append(valid_loss)
        log["train_batch_loss"].extend(train_losses)
        log["valid_batch_loss"].extend(valid_losses)
        log["best_epoch"] = best_epoch
        log["best_val_loss"] = best_val_loss
        log["last_epoch"] = epoch + 1
        
        with open(os.path.join(log_dir, "log.json"), "w") as f:
            json.dump(log, f)

        print(f"---- Epoch {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f} | Best Valid loss: {best_val_loss:.4f} | Best epoch: {best_epoch}")
    return log


def main():
    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])
    
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
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optim = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)
    
    # load dữ liệu
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_data=train_src_data,
        target_data=train_trg_data,
        source_max_length=configs["source_max_length"],
        target_max_seq_len=configs["target_max_seq_len"]
    )
    valid_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=valid_src_data, 
        target_data=valid_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )
    
    device = torch.device(configs["device"])
    
    # data loader bằng pytorch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    
    valid_loader =  torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=False
    )
    
    model.to(configs["device"])
    train(model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        n_epochs=configs["n_epochs"],
        target_pad_id=target_tokenizer.pad_token_id,
        device=device,
        model_path=configs["model_path"],
        early_stopping=configs["early_stopping"]
    )
    
    plot_loss(log_path="./logs/log.json", log_dir="./logs")
    
if __name__ == "__main__":
    main()