import torch
import json
import os
import matplotlib.pyplot as plt

# cấu hình 1 số tham số 
configs = {
    "train_soure_data" : "../../data/data_train/train.en",
    "train_target_data" : "../../data/data_train/train.vi",
    "valid_source_data" : "../../data/data_test/2013/tst2013.en",
    "valid_target_data" : "../../data/data_test/2013/tst2013.vi",
    "source_tokenizer":"bert-base-uncased",
    "target_tokenizer":"vinai/phobert-base",
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "max_seq_len":256,
    "batch_size":20,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "hidden_dim": 2048,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.1,
    "lr":0.0001,
    "n_epochs":50,
    "print_freq": 5,
    "beam_size":3,
    "model_path":"../save_model/model_transformer_translate_en_vi.pt",
    "early_stopping":5
}

# visualize log
def plot_loss(log_path, log_dir):
    log = json.load(open(log_path, "r"))

    plt.figure()
    plt.plot(log["train_loss"], label="train loss")
    plt.plot(log["valid_loss"], label="valid loss")
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_epoch.png"))

    # plot batch loss
    plt.figure()
    lst = log["train_batch_loss"]
    n = int(len(log["train_batch_loss"]) / len(log["valid_batch_loss"]))
    train_batch_loss = [lst[i:i + n][0] for i in range(0, len(lst), n)]
    plt.plot(train_batch_loss, label="train loss")
    plt.plot(log["valid_batch_loss"], label="valid loss")
    plt.title("Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_batch.png"))