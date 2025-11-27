
import os, sys, subprocess, json
import numpy as np

from setup import bioiain, bi, config
from bioiain.biopython.DSSP import ss_to_index

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ResidueDataset(Dataset):
    def __init__(self, struc_list, folder, label_folder=None):
        self.structures = struc_list
        self.folder = folder
        if label_folder is None:
            self.label_folder = folder
        else:
            self.label_folder = label_folder
        # self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.long)
        self.current_s = None
        self.current_e = None
        self.current_l = None
        self.pointer = {}
        self.total = 0
        for s in self.structures:
            code, ch = s.split("_")
            lp = os.path.join(self.label_folder, f"{code}.labels.json")
            lj = json.load(open(lp))

            for i, _ in enumerate(lj[ch].keys()):
                self.pointer[self.total] = {"s": s, "i": i} # {id_ch}, "res"
                self.total += 1

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        #print(self.pointer[idx])
        s, i = self.pointer[idx]["s"], self.pointer[idx]["i"]

        if s == self.current_s:
            embeddings = self.current_e
            labs = self.current_l
        else:
            code, ch = s.split("_")
            e_path = os.path.join(self.folder, f"{s}.pt")
            l_path = os.path.join(self.label_folder, f"{code}.labels.json")

            embeddings = torch.load(e_path)[0]
            label_json = json.load(open(l_path))[ch]
            labs = torch.tensor(np.array([r["label"] for r in label_json.values()]), dtype=torch.long)
            #print(labs)
            self.current_s = s
            self.current_e = embeddings
            self.current_l = labs



        #print(embeddings.shape)
        #print(labs.shape)

        emb = embeddings[i]
        lab = labs[i]

        #print("emb:", emb.shape, "lab:", lab)
        return emb, lab


def split_sample(array, folder, label_folder=None, test_ratio=0.2):

    bi.log(1, "splitting...")
    # Split into train/test
    train_list, test_list = train_test_split(array, test_size=test_ratio, random_state=42)

    bi.log(2, "Mounting data...")
    train_dataset = ResidueDataset(train_list, folder=folder, label_folder=label_folder)
    test_dataset = ResidueDataset(test_list, folder=folder, label_folder=label_folder)

    bi.log(3, "loading seq data...")
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=24, num_workers=0)

    return train_loader, test_loader, train_dataset, test_dataset







def train_mlp(model, train_loader, test_loader, lr=1e-3, epochs=20):
    bi.log(1, "Training MLP...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    for epoch in range(epochs):
        bi.log("header", f"Epoch: {epoch}")
        bi.log(1, "Training...")
        model.train()
        n_batches = len(train_loader)
        bi.log(2, "N batches:", n_batches)
        for n, (X_batch, y_batch) in enumerate(train_loader):
            bi.log(3, f"Batch: {n}/{n_batches}", end="\r")
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate
        bi.log(1, "Evaluating...")
        model.eval()
        preds = []
        labels_eval = []
        n_batches = len(test_loader)
        bi.log(2, "N batches:", n_batches)
        with torch.no_grad():
            for n, (X_batch, y_batch) in enumerate(test_loader):
                bi.log(3, f"Batch: {n}/{n_batches}", end="\r")
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                pred = logits.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                labels_eval.extend(y_batch.numpy())
        acc = accuracy_score(labels_eval, preds)
        if acc > best_acc:
            best_acc = acc
        bi.log(1, f"Test accuracy: {acc:.3f}")
        #bi.log("end", f"Epoch: {epoch}")
    return model, preds, labels_eval, best_acc





