
import os, sys, subprocess
import json
import numpy as np


import setup
setup.init()
from setup import bioiain, bi, config, log_file

from bioiain.biopython.DSSP import index_to_ss
from embeddings import generate_embeddings



bi.log("start", "internship > main.py")



skip_download = "--no-download" in sys.argv
force = "-f" in sys.argv
run_labels = "-l" in sys.argv
run_embeddings = "-e" in sys.argv
train = "-t" in sys.argv
predict = "-p" in sys.argv
curate = not("--no-curate" in sys.argv)
if "--all" in sys.argv:
    run_labels = True
    run_embeddings = True
    train = True

config["general"]["force"] = force
bi.log("header", "FORCE:", force)

with open(f"logs/{log_file}", "a") as log:
    log.write(f"""

### ### ARGVS

    Force: {force}

    Download: {not skip_download}
    Labels: {run_labels}
    Embeddings: {run_embeddings}
    Training: {train}
        Curate: {curate}
    Predict: {not predict}

###
""")


np.random.seed(config["general"]["np_random"])


data_folder = None
if not predict:
    bi.log("start", "Data Load")

    try:
        data_folder=config["data"]["selected"]["folder_name"]
        pdb_list_file = config["data"]["selected"]["pdb_list"]
    except:
        bi.log("error", "Dataset not configured")
        exit()

    if skip_download:
        file_folder = f"data/{data_folder}"
    else:
        file_folder = bi.biopython.downloadPDB("./data", data_folder, file_path=pdb_list_file, file_format="cif", overwrite=False)

    bi.log("header", "File folder:", file_folder)

    structure_list = sorted([f for f in os.listdir(file_folder) if ".swp" not in f])
    bi.log("end", "Data Load")

    with open(f"logs/{log_file}", "a") as log:
        log.write(f"""
        
### ### DATA
        
    N files: {len(structure_list)}
    Folder: {os.path.abspath(file_folder)}

### DATA CONFIG

{json.dumps(config["data"]["selected"], indent=2)}
        
### STRUCTURE LIST
        
{", ".join([os.path.basename(s) for s in structure_list])}
        
###
""")


if run_labels:
    bi.log("start", "LABELS")
    label_folder = config["labels"]["selected"]["save_folder"]
    bi.log(2, "Label folder:", label_folder)
    for file in structure_list:
        name = file.split(".")[0]

        structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{name}.cif"))
        bi.log("header", "Structure:",  structure)

        last_chain = [c.id for c in structure.get_chains()][-1]
        bi.log(1, "Last Chain:", last_chain)



        #print(os.path.exists(os.path.join(label_folder, f"{name}.labels.json")), force)
        #print(os.path.join(label_folder, f"{name}_{last_chain}.labels.json"))
        if not (os.path.exists(os.path.join(label_folder, f"{name}.labels.json"))) or force:
            bi.log(2, "Generating Labels...")
            from labels import generate_labels
            if not generate_labels(name, structure):
                continue
    bi.log("end", "LABELS")
    with open(f"logs/{log_file}", "a") as log:
        log.write(f"""

### ### LABELS

    N labels: {len(os.listdir(label_folder))}
    Folder: {os.path.abspath(label_folder)}

### LABEL CONFIG

{json.dumps(config["labels"]["selected"], indent=2)}

""")


if run_embeddings:
    bi.log("start", "EMBEDDINGS")
    embedding_folder = config["embeddings"]["selected"]["save_folder"]
    bi.log(1, "Embedding Folder:", embedding_folder)
    for file in structure_list:
        name = file.split(".")[0]

        structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{name}.cif"))
        bi.log("header", "Structure:", structure)

        last_chain = [c.id for c in structure.get_chains()][-1]
        bi.log(1, "Last Chain:", last_chain)


        #print(os.path.exists(os.path.join(embedding_folder, f"{name}_{last_chain}.pt")))
        #print(os.path.join(embedding_folder, f"{name}_{last_chain}.pt"))
        if not (os.path.exists(os.path.join(embedding_folder, f"{name}_{last_chain}.pt"))) or force:
            from embeddings import generate_embeddings
            generate_embeddings(name, structure)
    bi.log("end", "EMBEDDINGS")
    with open(f"logs/{log_file}", "a") as log:
        log.write(f"""

### ### EMBEDDINGS
    
    N embeddings: {len(os.listdir(embedding_folder))}
    Folder: {os.path.abspath(embedding_folder)}

### EMBEDDING CONFIG

{json.dumps(config["embeddings"]["selected"], indent=2)}

###
""")



if train:
    bi.log("start", "Model Training")
    import torch
    from models import get_model_class
    from training import train_mlp, split_sample
    from plotting import plot_confusion, plot_embeddings


    try:
        model_name=config["training"]["selected"]["model"]
    except:
        bi.log("error", "Training settings not configured")
        exit()

    bi.log(1, "Model name:", model_name)


    model_class = get_model_class(model_name)
    bi.log(2, model_class)

    try:
        config["training"]["selected_embedding"] = config["embeddings"]["selected"]
        embedding_folder =  config["training"]["selected_embedding"]["save_folder"]
    except:
        bi.log("error", "Embedding settings not configured")
        exit()

    bi.log(2, "Embedding folder:", embedding_folder)
    try:
        config["training"]["selected_label"] = config["labels"]["selected"]
        label_folder =  config["training"]["selected_label"]["save_folder"]
    except:
        bi.log("error",f"Label settings not configured")
        exit()

    bi.log(2, "Label folder:", label_folder)



    training_structures = []
    bi.log(1, "Curating input...")
    file_n = len(os.listdir(embedding_folder))
    for n, file in enumerate(sorted(os.listdir(embedding_folder), reverse=True)):
        if not(file.split("_")[0]+".cif" in structure_list):
            continue
        fname = file.split(".")[0]
        name, ch = fname.split("_")

        bi.log(2, file, fname, end="\r")
        #print(file.endswith(".pt"), f"{label_folder}/{name}.labels.json")
        if file.endswith(".pt") and os.path.exists(f"{label_folder}/{name}.labels.json"):
            if curate:
                embedding = torch.load(f"{embedding_folder}/{fname}.pt")[0][1:-1]
                labels = json.load(open(f"{label_folder}/{name}.labels.json"))

                #print(embedding.shape[0])
                #print(len(labels[ch]))
                if embedding.shape[0] != len(labels[ch]):
                    bi.log("warning", "Embeddings and labels do not match:", ch, embedding.shape[0], len(labels[ch]))
                    # os.remove(os.pat.join(embedding_folder, file))
                    continue

            training_structures.append(fname)

    #print(training_structures[0:3], "...", training_structures[-3:])
    num_structs = len(training_structures)
    bi.log(2, f"Training with {num_structs}")
    bi.log(3, "Number of PDB codes:", len(set([c[:5] for c in training_structures])))
    bi.log(3, f"Number of available structures: {file_n}" )
    bi.log(1, "Input curated")


    bi.log("header", "Training MLP...")
    bi.log(1, "Model name:", model_name)
    bi.log(2, model_class)

    train_loader, test_loader, train_dataset, test_dataset = split_sample(
        training_structures, embedding_folder, label_folder, config["training"]["selected"]["test_ratio"])

    os.makedirs("models", exist_ok=True)

    title = f"{model_name}_{data_folder}_{config["labels"]["selected"]["method"]}_{config['embeddings']['selected']['model']}"
    classes = config["labels"]["selected"]["classes"]
    print(classes)
    mlp = model_class(input_dim=config["training"]["selected_embedding"]["dimensions"], num_classes=len(classes))

    mlp, preds, labels, score = train_mlp(mlp, train_loader, test_loader, epochs=config["training"]["selected"]["epochs"])
    torch.save(mlp.state_dict(), f"models/{title}.pth")

    bi.log("header", "Plotting results...")
    os.makedirs("figs", exist_ok=True)

    plot_embeddings(test_dataset, labels, title=title, score = score)
    plot_confusion(preds, labels, title=title, score=score, classes=classes)
    bi.log("header", "DONE")

    bi.log("end", "Model Training")




if predict:
    bi.log("start", "Predicting...")

    from models import get_model_class



    model_class = get_model_class(config["predict"]["selected"]["model"])
    bi.log("header", model_class)

    import Bio


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









