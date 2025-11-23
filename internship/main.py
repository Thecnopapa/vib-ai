
import os, sys, subprocess

import json
import numpy as np
import torch

import Bio


try:
    raise ImportError("bioiain")
except:
    try:
        import importlib
        sys.path.append("/home/iain/projects/bioiain")
        import src.bioiain as bi
        bioiain = bi
    except:
        raise ImportError("bioiain")
print(bi)
print(bioiain)
from bioiain.utilities.DSSP import ss_to_index, index_to_ss
from embeddings import run_dssp, run_foldseek, generate_embeddings



bi.log("start", "internship > main.py")


skip_download = "no-download" in sys.argv
force = "force" in sys.argv
embeddings = "embeddings" in sys.argv
train = "train" in sys.argv
predict = "predict" in sys.argv

np.random.seed(0)
num_residues = 255
embedding_dim = 480

if not predict:
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

if (force or embeddings) and not predict:

    for file in structure_list:

        name = file.split(".")[0]

        structure = None
        #structure = bi.imports.recover("1MOT")

        structure = bi.imports.loadPDB(os.path.join(file_folder, f"{name}.cif"))
        structure.pass_down()
        bi.log("header", structure)



        last_chain = [c.id for c in structure.get_chains()][-1]
        if not os.path.exists(f"./out/SaProt/full/{name}_{last_chain}.pt") or force:
            dssp_dict = run_dssp(structure, name, "data/"+data_folder)
            print(dssp_dict.keys())
            dssp_dict = run_foldseek(structure, name, dssp_dict, "data/"+data_folder)
            embedding_folder = generate_embeddings(dssp_dict, name)
            print(dssp_dict.keys())




if force or train:

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sb



    curate = not("no-curate" in sys.argv)
    training_structures = []
    bi.log("start", "Curating embeddings...")
    file_n = len(os.listdir("out/SaProt/full"))
    for n, file in enumerate(sorted(os.listdir("out/SaProt/full"), reverse=True)):
        if not(file.split("_")[0]+".cif" in structure_list):
            continue
        fname = file.split(".")[0]
        if file.endswith(".pt") and os.path.exists(f"out/SaProt/full/{fname}.json") and os.path.exists(f"out/SaProt/seq_only/{fname}.pt"):
            if curate:
                emb_struc = torch.load(f"out/SaProt/full/{fname}.pt")
                reslist = json.load(open(f"out/SaProt/full/{fname}.json"))
                emb_seq = torch.load(f"out/SaProt/seq_only/{fname}.pt")

                new_struc = emb_struc[0][1:-1].tolist()
                new_seq = emb_seq[0][1:-1].tolist()
                new_lab = []
                for ress in reslist.values():
                    # print(ress)
                    try:
                        int(ress["res"])
                        new_lab.append(ss_to_index(ress["ss"]))
                    except:
                        #bi.log("warning", "Disordered res:", ress["res"])
                        new_lab.append(ss_to_index(ress["ss"]))
                print(len(new_struc), "\t", len(new_seq), "\t", len(new_lab), "\t", fname, f"\t{n}/{file_n}", end="\r")

                try:
                    assert (len(new_struc) == len(new_seq) == len(new_lab))
                    training_structures.append(fname)
                except:
                    f = f"out/SaProt/full/{fname}.pt"
                    os.remove(f)
                    bi.log("error", "Missmatch in file (removed):", f)
                    continue
            training_structures.append(fname)

    print(training_structures[0:3], "...", training_structures[-3:])
    num_structs = len(training_structures)
    print("Training structures:", num_structs)
    bi.log("end", "Curating embeddings")



    class ResidueDataset(Dataset):
        def __init__(self, struc_list, folder, label_folder=None):
            self.structures = struc_list
            self.folder = folder
            if label_folder is None:
                self.label_folder = folder
            else:
                self.label_folder = label_folder
            #self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            #self.labels = torch.tensor(labels, dtype=torch.long)
            self.current_s = None
            self.current_e = None
            self.current_l = None
            self.pointer = {}
            self.total = 0
            for s in self.structures:
                lp = os.path.join(self.label_folder, f"{s}.json")
                lj = json.load(open(lp))

                for i, _ in enumerate(lj.keys()):
                    self.pointer[self.total] = {"s":s, "i":i}
                    self.total += 1

        def __len__(self):
            return self.total

        def __getitem__(self, idx):
            s, i = self.pointer[idx]["s"], self.pointer[idx]["i"]
            if s == self.current_s:
                embeddings = self.current_e
                labs = self.current_l
            else:
                embedding_path = os.path.join(self.folder, f"{s}.pt")
                label_path = os.path.join(self.label_folder, f"{s}.json")

                embeddings = torch.load(embedding_path)[0][1:-1]
                label_json = json.load(open(label_path))
                labs = torch.tensor(np.array([ss_to_index(r["ss"]) for r in label_json.values()]), dtype=torch.long)

                self.current_s = s
                self.current_e = embeddings
                self.current_l = labs

            #print(embeddings.shape)
            #print(labs.shape)

            emb = embeddings[i]
            lab = labs[i]
            #print("emb:", emb.shape, "lab:", lab)
            return emb, lab





    bi.log("start", "splitting...")
    # Split into train/test
    train_list, test_list = train_test_split(training_structures, test_size=0.2, random_state=42)

    bi.log("start", "mounting seq data...")
    train_dataset_seq = ResidueDataset(train_list, folder="out/SaProt/seq_only", label_folder="out/SaProt/full")
    test_dataset_seq = ResidueDataset(test_list, folder="out/SaProt/seq_only", label_folder="out/SaProt/full")
    bi.log("start", "mounting struc data...")
    train_dataset_struct = ResidueDataset(train_list, folder="out/SaProt/full")
    test_dataset_struct = ResidueDataset(test_list, folder="out/SaProt/full")

    bi.log("start", "loading seq data...")
    train_loader_seq = DataLoader(train_dataset_seq, batch_size=24, shuffle=True, num_workers=0)
    test_loader_seq = DataLoader(test_dataset_seq, batch_size=24, num_workers=0)
    bi.log("start", "loading struc data...")
    train_loader_struct = DataLoader(train_dataset_struct, batch_size=24, shuffle=True, num_workers=0)
    test_loader_struct = DataLoader(test_dataset_struct, batch_size=24, num_workers=0)
    bi.log("end")




    # ---------------------------
    # Training function
    # ---------------------------
    def train_mlp(model, train_loader, test_loader, lr=1e-3, epochs=20):
        print("training...")
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


    from models import MLP
    os.makedirs("models", exist_ok=True)
    # ---------------------------
    # Train MLP on sequence-only embeddings
    # ---------------------------
    bi.log("start", "Training MLP on sequence-only embeddings...")
    mlp_seq = MLP(input_dim=embedding_dim)
    mlp_seq, preds_seq, labels_seq = train_mlp(mlp_seq, train_loader_seq, test_loader_seq)
    torch.save(mlp_seq.state_dict(), f"models/{data_folder}_seq.pth")
    bi.log("end", "Training MLP on sequence+3Di embeddings")

    # ---------------------------
    # Train MLP on structure-aware embeddings
    # ---------------------------
    bi.log("start", "Training MLP on sequence+3Di embeddings...")
    mlp_struct = MLP(input_dim=embedding_dim)
    mlp_struct, preds_struct, labels_struct = train_mlp(mlp_struct, train_loader_struct, test_loader_struct)
    torch.save(mlp_seq.state_dict(), f"models/{data_folder}_struct.pth")
    bi.log("end", "Training MLP on sequence+3Di embeddings")

    bi.log("start", "Plotting...")
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

    plot_embeddings(test_dataset_seq, labels_seq, title=f"{data_folder}_sequence_only_N_{len(training_structures)}")
    plot_embeddings(test_dataset_struct, labels_struct, title=f"{data_folder}_sequence_+_3Di_N_{len(training_structures)}")

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


    bi.log("end", "Plotting")








if predict:
    bi.log("start", "Predicting...")

    try:
        seq_index = sys.argv.index("-s") + 1
        sequence = sys.argv[seq_index]
    except Exception as e:
        bi.log("error", "no input sequence provided (-s)")
        sequence = bi.utilities.strings.clean_string(input("Please enter a sequence to predict: \n>>> ").replace(" ", ""))
    if len(sequence) == 0:
        bi.log("error", "No input sequence provided")
        exit()


    os.makedirs("pred", exist_ok=True)


    n = 0
    fname = f"prediction_{bi.utilities.strings.add_front_0(n, 3)}"
    while os.path.exists(f"pred/{fname}"):
        n += 1
        fname = f"prediction_{bi.utilities.strings.add_front_0(n, 3)}"

    os.makedirs(f"pred/{fname}", exist_ok=True)

    structure = None
    try:
        pdb_index = sys.argv.index("-pdb") + 1
        pdb_code = sys.argv[pdb_index]
    except Exception as e:
        bi.log("error", "no pdb code or file provided (-pdb)")
        pdb_code = bi.utilities.strings.clean_string(input("Please enter a pdb code or file path to compare: \n>>> ").strip())
    if len(pdb_code) == 0:
        bi.log("error", "No input pdb provided")
    elif len(pdb_code) == 4:
        download_folder = bi.imports.downloadPDB("pred", fname, [pdb_code], file_format="cif")
        file_path = os.path.join(download_folder, f"{pdb_code}.cif")
        structure = bi.imports.loadPDB(file_path)
    else:
        if os.path.exists(pdb_code):
            structure = bi.imports.loadPDB(pdb_code)
        else:
            bi.log("error", "Provided path does not exist: {}".format(pdb_code))
            exit()
    print(structure)
    bi.log("header", "Sequence:")
    dssp = {"A":{str(n):{
        "resn":name,
        "ss": "#",
        "res":n,
    } for n, name in enumerate(sequence)}}
    print("".join([r["resn"] for r in dssp["A"].values()]))
    print("length:", len(dssp["A"]))


    bi.log(1, "Assigned id:", fname)
    json.dump({
        "sequence": "".join([r["resn"] for r in dssp["A"].values()]),
        "dssp": dssp}, open(f"pred/{fname}/{fname}.json", "w"))

    #print(dssp)
    generate_embeddings(dssp, fname, folder=f"pred/{fname}/SaProt", no3D=True)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = "models/mega-batch_struct.pth"
    model = MLP(input_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, weights_only=False))

    model.eval()
    preds = []
    labels_eval = []


    embeddings = torch.load(f"pred/{fname}/SaProt/seq_only/{fname}_A.pt")[0][2:-2]
    bi.log(1, "n embeddings:", len(embeddings), embeddings.shape)
    with torch.no_grad():
        for X_batch in embeddings:
            X_batch = X_batch.to(device)
            logits = list(model(X_batch).numpy())
            #print(logits)
            pred = logits.index(max(logits))
            preds.append(pred)
    bi.log("header", "Predictions:")
    print("".join([str(p) for p in preds]))
    print("len:", len(preds))

    if structure is not None:
        for chain in structure.get_chains():
            print(chain)
            for res, p in zip(chain.get_residues(), preds):
                print([a for a in res])
                try:
                    ca = [a for a in res if a.id == "CA"][0]
                    ca.bfactor = int(p)
                except:
                    pass

    dssp = run_dssp(structure, structure.data["info"]["name"], data_folder=f"pred/{fname}", out_folder=f"pred/{fname}")
    matching_chain = "A"
    for ch in dssp.keys():
        print(ch, len(dssp[ch]), len(preds))
        if len(dssp[ch]) == len(preds):
            matching_chain = ch
    aligner = Bio.Align.PairwiseAligner()
    print(aligner)
    al = aligner.align("".join([d["ss"].replace("-", "#") for d in dssp[matching_chain].values()]), "".join([index_to_ss(p) for p in preds]),)
    print(al[0])
    with open(f"pred/{fname}/output.txt", "w") as f:
        f.write("SEQUENCE>\t{}\n".format("".join([r["resn"] for r in dssp["A"].values()])))
        f.write("PREDICT >\t{}\n".format("".join([index_to_ss(p) for p in preds])))
        if matching_chain is not None:
            f.write("DSSP    >\t{}\n".format("".join([d["ss"] for d in dssp[matching_chain].values()])))
        f.write(f"\n\n\nAlignment (score: {al[0].score})/{len(al[0].query)}\n")
        f.write(str(al[0]))




    session = bi.visualisation.pymol.PymolScript(fname, f"pred/{fname}/session")
    session.load_entity(structure)
    session.spectrum("(all)", "b", "rainbow", minimum=0, maximum=7)
    session.write_script()



    bi.log("end")









