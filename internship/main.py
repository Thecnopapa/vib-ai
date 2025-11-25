
import os, sys, subprocess

import json
import numpy as np


import setup
setup.init()
from setup import bioiain, bi, config

from bioiain.biopython.DSSP import ss_to_index, index_to_ss
from embeddings import run_foldseek, generate_embeddings



bi.log("start", "internship > main.py")


skip_download = "--no-download" in sys.argv
force = "-f" in sys.argv
embeddings = "-e" in sys.argv
train = "-t" in sys.argv
predict = "-p" in sys.argv
curate = not("--no-curate" in sys.argv)

config["general"]["force"] = force
bi.log("header", "FORCE:", force)


np.random.seed(config["general"]["np_random"])



if not predict:
    bi.log("start", "Data Load")
    if "--data" in sys.argv:
        try:
            dataset = sys.argv[sys.argv.index("--data") + 1]
        except:
            bi.log("error", "Dataset not provided")
            exit()
    else:
        dataset = config["data"]["default"]
    bi.log("header", "Using dataset:", dataset)
    config["data"]["selected"] = config["data"][dataset]
    try:
        data_folder=config["data"][dataset]["folder_name"]
        pdb_list_file = config["data"][dataset]["pdb_list"]
    except:
        bi.log("error", f"Dataset not configured: {dataset}")
        exit()

    if skip_download:
        file_folder = f"data/{data_folder}"
    else:
        file_folder = bi.biopython.downloadPDB("./data", data_folder, file_path=pdb_list_file, file_format="cif", overwrite=False)

    bi.log("header", "File folder:", file_folder)

    structure_list = sorted(os.listdir(file_folder))
    bi.log("end", "Data Load")

if embeddings:
    bi.log("start", "Labels and Embeddings")
    for file in structure_list:
        name = file.split(".")[0]

        structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{name}.cif"))
        bi.log("header", "Structure:",  structure)

        last_chain = [c.id for c in structure.get_chains()][-1]
        bi.log(1, "Last Chain:", last_chain)

        if "--labels" in sys.argv:
            label_method = sys.argv[sys.argv.index("--labels") + 1]
        else:
            label_method = config["labels"]["default"]
        bi.log(1, "Label method:", label_method)
        config["labels"]["selected"] = config["labels"][label_method]
        label_folder = config["labels"][label_method]["save_folder"]
        bi.log(2, "Label folder:", label_folder)
        #print(os.path.exists(os.path.join(label_folder, f"{name}.labels.json")), force)
        #print(os.path.join(label_folder, f"{name}_{last_chain}.labels.json"))
        if not (os.path.exists(os.path.join(label_folder, f"{name}.labels.json"))) or force:
            bi.log(2, "Generating Labels...")
            from labels import generate_labels
            if not generate_labels(name, structure):
                continue

        if "--embedding" in sys.argv:
            embedding_method = sys.argv[sys.argv.index("--embedding") + 1]
        else:
            embedding_method = config["embeddings"]["default"]
        bi.log(1, "Embedding method:", embedding_method)
        config["embeddings"]["selected"] = config["embeddings"][embedding_method]
        embedding_folder = config["embeddings"][embedding_method]["save_folder"]
        bi.log(2, "Embedding Folder:", embedding_folder)
        #print(os.path.exists(os.path.join(embedding_folder, f"{name}_{last_chain}.pt")))
        #print(os.path.join(embedding_folder, f"{name}_{last_chain}.pt"))
        if not (os.path.exists(os.path.join(embedding_folder, f"{name}_{last_chain}.pt"))) or force:
            generate_embeddings(name, structure)
    bi.log("end", "Labels and Embeddings")



if train:

    import torch
    from training import train_mlp, split_sample
    from plotting import plot_confusion, plot_embeddings
    from models import MLP




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


    config = {
        "seq": {
            "folder": "out/SaProt/seq_only",
            "label_folder": "out/SaProt/full",
            "active": False,
            "epochs": 20,
            "test_ratio": 0.2,
        },
        "struct": {
            "folder": "out/SaProt/full",
            "label_folder": None,
            "active": True,
            "epochs": 2,
            "test_ratio": 0.2,
        }
    }

    for k, v in config.items():
        if v["active"]:
            bi.log("start", f"Training MLP for: {k}")

            train_loader, test_loader, train_dataset, test_dataset = split_sample(training_structures, v["folder"], v["label_folder"], test_ratio=v["test_ratio"])

            os.makedirs("models", exist_ok=True)

            mlp = MLP(input_dim=embedding_dim)
            mlp, preds, labels, score = train_mlp(mlp, train_loader, test_loader, epochs=v["epochs"])
            torch.save(mlp.state_dict(), f"models/{data_folder}_{k}_N={len(training_structures)}_E={v["epochs"]}_S={score:.3f}.pth")

            bi.log("header", "Plotting...")
            os.makedirs("figs", exist_ok=True)
            plot_embeddings(test_dataset, labels,
                            title=f"{data_folder}_{k}_{len(training_structures)}_E={v["epochs"]}_S={score:.3f}")
            plot_confusion(preds, labels, title=f"{data_folder}_{k}_N={len(training_structures)}_E={v["epochs"]}_S={score:.3f}")
            bi.log("header", "DONE")
            bi.log("end", f"Training MLP for {k}")




if predict:
    from models import MLP

    import Bio
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
    device = "cpu"

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









