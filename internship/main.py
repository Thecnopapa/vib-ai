
import os, sys, subprocess
from copy import deepcopy
import json
import pandas as pd

from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from networkx.algorithms.centrality.flow_matrix import flow_matrix_row

sys.path.append("/home/iain/projects/bioiain")

import src.bioiain as bi
from src.bioiain.utilities.DSSP import DSSP, ss_to_index


bi.log("start", "internship > main.py")
skip_download = "no-download" in sys.argv
force = "force" in sys.argv
embeddings = "embeddings" in sys.argv
train = "train" in sys.argv
data_folder="receptors"
pdb_list_file = "data/receptors.txt"
if "mega" in sys.argv:
    data_folder = "mega-batch"
    pdb_list_file = "data/mega-batch20K.txt"

if skip_download:
    file_folder = f"data/{data_folder}"
else:
    file_folder = bi.imports.downloadPDB("./data", data_folder, file_path=pdb_list_file, file_format="cif", overwrite=False)

bi.log(1, "File folder:", file_folder)

structure_list = sorted(os.listdir(file_folder))

if force or embeddings:

    for file in structure_list:

        name = file.split(".")[0]

        structure = None
        #structure = bi.imports.recover("1MOT")


        bi.log(1, "Structure:", structure)


        if structure is None:
            pass
        structure = bi.imports.loadPDB(os.path.join(file_folder, f"{name}.cif"))
        structure.pass_down()
        #bi.log(1, "Structure:", structure)
        bi.log("header", structure)

        #structure.export(structure_format="cif")
        model = structure[0]
        bi.log(1, "Model:", model)
        model.__getitem__ = model._getitem
        model.export()
        #print(model.__getitem__)
        print(model.get_list())
        #print(model.child_dict)
        #print(model[0])

        # p = PDBParser()
        # structure = p.get_structure("1MOT", "data/other/1MOT.cif")
        # model = structure[0]
        # print(DSSP)
        # dssp_dict = dssp_dict_from_pdb_file("data/other/1MOT.pdb")
        # print(dssp_dict)

        def run_dssp(structure, filename):
            os.makedirs("out", exist_ok=True)
            os.makedirs("out/dssp", exist_ok=True)
            "dssp --output-format dssp ./data/{data_folder}/{filename}.cif ./out/dssp/{filename}.dssp"
            cmd = ["dssp", "--output-format", "dssp", "--verbose", f"./data/{data_folder}/{filename}.cif", f"./out/dssp/{filename}.dssp"]
            subprocess.run(cmd)
            #return DSSP(model, f"./data/other/{filename}.cif", file_type="DSSP")
            dssp_dict = {c.id:{} for c in structure.get_chains()}
            #print(dssp_dict)
            with open(f"./out/dssp/{filename}.dssp", "r") as f:
                start = False

                for line in f:

                    if "#" in line:
                        start = True
                        continue
                    if not start:
                        continue
                    l = line.split(" ")
                    l = [cl for cl in l if cl != ""]

                    #res = l[1]
                    #ch = l[2]
                    #resn = l[3]
                    #ss = l[4]
                    if "!" in line:
                        continue
                    res = line[5:11].strip()
                    ch = line[11]
                    resn = line[13].upper()
                    ss = line[16]
                    if line[10] != " ":
                        bi.log("warning", f"Disordered atom: \n {line}")
                        #exit()
                        #continue

                    #print(ch, res, resn, ss)


                    dssp_dict[ch][res] = {"res": res, "resn": resn, "ss": ss}

            return dssp_dict






        def run_foldseek(structure, filename, dssp_dict):
            bi.log(2, "running foldseek...")
            "./SaProt/bin/foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 ./data/other/1M2Z.cif ./out/foldseek/1M2Z.tsv"
            os.makedirs("out", exist_ok=True)
            os.makedirs("out/foldseek", exist_ok=True)
            cmd = ["./SaProt/bin/foldseek",
                   "structureto3didescriptor", "-v", "0", "--threads", "4",
                   "--chain-name-mode", "1", f"./data/{data_folder}/{filename}.cif",
                   f"./out/foldseek/{filename}.csv"
            ]
            print(" ".join(cmd))
            subprocess.run(cmd)
            with open(f"./out/foldseek/{filename}.csv", "r", encoding="utf-8") as f:
                for line in f:
                    ch = line.split("\t")[0].split("_")[1].split(" ")[0]
                    bi.log(3, "foldseek out:", ch )
                    l = line.split("\t")[2]
                    toks = [t for t in l]
                    print(len(dssp_dict[ch].items()), len(toks))
                    if not len(dssp_dict[ch].items()) == len(toks):
                        dssp_dict.pop(ch)
                        bi.log("warning", "foldseek tokens and dssp_dict do not match:", ch )
                        continue
                    for n, (k, v) in enumerate(dssp_dict[ch].items()):
                        if toks[n] == " ":
                            toks = "-"
                        dssp_dict[ch][k]["fs"] = toks[n]
            return dssp_dict


        def generate_embeddings(dssp_dict):
            bi.log(2, "Generating embeddings...")
            seqs = {k: [] for k in dssp_dict.keys()}
            seqs3D = {k: [] for k in dssp_dict.keys()}
            for k, v in dssp_dict.items():
                #print(v)
                #print(k)
                try:
                    seqs3D[k] = ["{}{}".format(i["resn"].upper(), i["fs"].lower()) for i in v.values()]
                except:
                    continue
                seqs[k] = ["{}{}".format(i["resn"].upper(), "#") for i in v.values()]
            for ch in seqs.keys():
                # Load model directly
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_35M_AF2")
                model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_35M_AF2")

                tokenizerS = AutoTokenizer.from_pretrained("westlake-repl/SaProt_35M_AF2_seqOnly")
                modelS = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_35M_AF2_seqOnly")

                # print(tokenizer)
                # print(model)

                modelS.eval()
                modelS.to(device)

                seq = "".join(seqs[ch])
                long_seq3D = "".join(seqs3D[ch])


                inputs = tokenizerS(seq, return_tensors="pt").to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                print(inputs)

                with torch.no_grad():
                    outputs = modelS(**inputs, output_hidden_states=True)
                # print(outputs)

                # outputs.hidden_states is a tuple of all layers, including embeddings
                # Shape of each layer: [batch_size, sequence_length, hidden_dim]
                all_hidden_states = outputs.hidden_states

                # Last layer hidden states
                last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
                print(last_hidden.shape)  # ['<cls>', 'M#', 'E#', 'V#', 'Q#', '<eos>']
                print(last_hidden)

                os.makedirs("out/SaProt/seq_only", exist_ok=True)
                torch.save(last_hidden, f"out/SaProt/seq_only/{name}_{ch}.pt")


                model.eval()
                model.to(device)

                inputs = tokenizer(long_seq3D, return_tensors="pt").to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # outputs.hidden_states is a tuple of all layers, including embeddings
                # Shape of each layer: [batch_size, sequence_length, hidden_dim]
                all_hidden_states = outputs.hidden_states

                # Last layer hidden states
                last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
                print(last_hidden.shape)
                print(last_hidden)
                os.makedirs("out/SaProt/full", exist_ok=True)
                torch.save(last_hidden, f"out/SaProt/full/{name}_{ch}.pt")
                json.dump(dssp_dict[ch], open(f"out/SaProt/full/{name}_{ch}.json", "w"))
            return dssp_dict


        last_chain = [c.id for c in structure.get_chains()][-1]
        if not os.path.exists(f"./out/SaProt/full/{name}_{last_chain}.pt") or force:
            dssp_dict = run_dssp(structure, name)
            print(dssp_dict.keys())
            dssp_dict = run_foldseek(structure, name, dssp_dict)
            dssp_dict = generate_embeddings(dssp_dict)
            print(dssp_dict.keys())




if force or train:

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sb
    import numpy as np




    all_residues_seq = []
    all_embeddings_seq = []

    all_residues_struc = []
    all_embeddings_struc = []
    num_sequences = 0
    num_structures = 0
    training_structures = []
    print(len(os.listdir("out/SaProt/seq_only")), len(os.listdir("out/SaProt/full"))/2)
    for file in sorted(os.listdir("out/SaProt/full"), reverse=True):
        if not(file.split("_")[0]+".cif" in structure_list):
            continue
        fname = file.split(".")[0]
        if file.endswith(".pt") and os.path.exists(f"out/SaProt/full/{fname}.json") and os.path.exists(f"out/SaProt/seq_only/{fname}.pt"):
            all_embeddings_struc.append(torch.load(f"out/SaProt/full/{fname}.pt"))
            all_residues_struc.append(json.load(open(f"out/SaProt/full/{fname}.json")))
            all_embeddings_seq.append(torch.load(f"out/SaProt/seq_only/{fname}.pt"))
            num_sequences += 1
            num_structures += 1
            training_structures.append(fname)
            #print(file)



    print("Num filels:", num_structures, num_sequences)

    np.random.seed(0)
    num_residues = 255
    embedding_dim = 480
    num_structs = 300

    bi.log("start")

    total_res = sum([len(r) for r in all_residues_seq])
    print(total_res)
    embeddings_struc = []
    embeddings_seq = []
    labels = []
    embs = zip(all_embeddings_struc[0:num_structs], all_embeddings_seq[0:num_structs], all_residues_struc[0:num_structs] )
    for n, (emb_struc, emb_seq, reslist) in enumerate(embs):
        embeddings_struc.extend(emb_struc[0][1:-1].tolist())
        embeddings_seq.extend(emb_seq[0][1:-1].tolist())
        for ress in reslist.values():
            #print(ress)
            try:
                int(ress["res"])
                labels.append(ss_to_index(ress["ss"]))
            except:
                bi.log("warning", "Disordered res:", ress["res"])
                labels.append(ss_to_index(ress["ss"]))
        print(len(embeddings_struc))
        print(len(embeddings_seq))
        print(len(labels))
        print(training_structures[n])
        try:
            assert(len(embeddings_struc) == len(embeddings_seq) == len(labels))
        except:
            f = f"out/SaProt/full/{training_structures[n]}.pt"
            os.remove(f)
            bi.log("error", "removed:", f)
            exit()

        bi.log("end")

        #labels.extend([ss_to_index(ss["ss"]) for ss in labs.values()])

    embeddings_struc = np.array(embeddings_struc)
    embeddings_seq = np.array(embeddings_seq)#
    labels = np.array(labels)
    print(len(embeddings_struc))
    print(len(embeddings_seq))
    print(len(labels))
    bi.log("end")




    emb_seq = embeddings_seq
    emb_struct = embeddings_struc
    #print(emb_seq[0])
    print(emb_seq.shape)
    print(emb_struct.shape)
    print(labels.shape)



    #print(emb_seq)
    #print("#")
    #print(emb_struct)
    #print(labels)



    class ResidueDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]


    # Split into train/test
    emb_train_seq, emb_test_seq, y_train, y_test = train_test_split(emb_seq, labels, test_size=0.2, random_state=42)
    emb_train_struct, emb_test_struct, _, _ = train_test_split(emb_struct, labels, test_size=0.2, random_state=42)

    train_dataset_seq = ResidueDataset(emb_train_seq, y_train)
    test_dataset_seq = ResidueDataset(emb_test_seq, y_test)

    train_dataset_struct = ResidueDataset(emb_train_struct, y_train)
    test_dataset_struct = ResidueDataset(emb_test_struct, y_test)

    train_loader_seq = DataLoader(train_dataset_seq, batch_size=32, shuffle=True)
    test_loader_seq = DataLoader(test_dataset_seq, batch_size=32)

    train_loader_struct = DataLoader(train_dataset_struct, batch_size=32, shuffle=True)
    test_loader_struct = DataLoader(test_dataset_struct, batch_size=32)

    # ---------------------------
    # MLP model
    # ---------------------------
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=[256,128], num_classes=8, dropout=0.2):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[1], num_classes)
            )
        def forward(self, x):
            return self.model(x)

    # ---------------------------
    # Training function
    # ---------------------------
    def train_mlp(model, train_loader, test_loader, lr=1e-3, epochs=20):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_acc = 0
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            preds = []
            labels_eval = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    logits = model(X_batch)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    preds.extend(pred)
                    labels_eval.extend(y_batch.numpy())
            acc = accuracy_score(labels_eval, preds)
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch+1}, Test accuracy: {acc:.3f}")
        return model, preds, labels_eval



    os.makedirs("models", exist_ok=True)
    # ---------------------------
    # Train MLP on sequence-only embeddings
    # ---------------------------
    print("Training MLP on sequence-only embeddings...")
    mlp_seq = MLP(input_dim=embedding_dim)
    mlp_seq, preds_seq, labels_seq = train_mlp(mlp_seq, train_loader_seq, test_loader_seq)
    torch.save(mlp_seq.state_dict(), f"models/{data_folder}_seq.pth")

    # ---------------------------
    # Train MLP on structure-aware embeddings
    # ---------------------------
    print("Training MLP on sequence+3Di embeddings...")
    mlp_struct = MLP(input_dim=embedding_dim)
    mlp_struct, preds_struct, labels_struct = train_mlp(mlp_struct, train_loader_struct, test_loader_struct)
    torch.save(mlp_seq.state_dict(), f"models/{data_folder}_struct.pth")


    bi.log("start", "Plotting...")
    0
    # ---------------------------
    # Visualization
    # ---------------------------
    os.makedirs("figs", exist_ok=True)
    def plot_embeddings(embeddings, labels, title="Embedding PCA"):
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(embeddings)
        plt.figure(figsize=(12,10))
        sb.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette="Set1", s=40, alpha=0.8)
        plt.title(title)
        plt.savefig(f"figs/embeddings_{title}.png")

    plot_embeddings(emb_test_seq, labels_seq, title=f"{data_folder}_sequence_only_N_{len(training_structures)}")
    plot_embeddings(emb_test_struct, labels_struct, title=f"{data_folder}_sequence_+_3Di_N_{len(training_structures)}")

    # Confusion matrices
    def plot_confusion(preds, labels, title="Confusion Matrix"):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8,8))
        sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['H','B','E','G',"I","T","S","-"], yticklabels=['H','B','E','G',"I","T","S","-"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.savefig(f"figs/confusion_{title}.png")

    plot_confusion(preds_seq, labels_seq, title=f"{data_folder}_sequence_only_N_{len(training_structures)}")
    plot_confusion(preds_struct, labels_struct, title=f"{data_folder}_sequence_+_3Di_N_{len(training_structures)}")