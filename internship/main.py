
import os, sys, subprocess
import json
import numpy as np
from matplotlib.style.core import available
from torch.xpu import device

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
terminal = True
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

    title = f"{model_name}_{data_folder}_{config['labels']['selected']['method']}_{config['embeddings']['selected']['model']}"
    classes = config["labels"]["selected"]["classes"]
    print(classes)
    mlp = model_class(input_dim=config["training"]["selected_embedding"]["dimensions"], num_classes=len(classes))

    mlp, preds, labels, score = train_mlp(mlp, train_loader, test_loader, epochs=config["training"]["selected"]["epochs"])
    torch.save(mlp.state_dict(), f"models/{title}.pth")
    model_info = {"labels": config["training"]["selected_label"],
                  "embeddings": config["training"]["selected_embedding"],
                  "training": config["training"]["selected"],
                  "data_folder": config["data"]["selected"]}
    json.dump(model_info, open(f"models/{title}.model.json", "w"), indent=4)

    bi.log("header", "Plotting results...")
    os.makedirs("figs", exist_ok=True)

    plot_embeddings(test_dataset, labels, title=title, score = score)
    plot_confusion(preds, labels, title=title, score=score, classes=classes)
    bi.log("header", "DONE")

    bi.log("end", "Model Training")




if predict:
    bi.log("start", "PREDICT")

    details = {}

    if terminal:
        bi.log("header", "Terminal mode")

        mode_ok = False
        available_models = [m.split(".")[0] for m in os.listdir("models") if m.endswith(".model.json")]
        while not mode_ok:
            bi.log("header", "Please select model to use:")
            [bi.log(1, f"{n}: {m}") for n, m in enumerate(available_models)]
            i = input("\n>>> ")

            try:
                model_name = available_models[int(i)]
                mode_ok = True
            except:
                bi.log("error", "Invalid model selected:", i)
        bi.log(1, "Model selected:", model_name)
        model_info = json.load(open(f"models/{model_name}.model.json"))
        details["model_name"] = model_name
        details["model_info"] = model_info

        sequence_ok = False
        while not sequence_ok:
            bi.log("header", "Please introduce sequence to predict")
            i = input("\n>>> ")

            try:

                sequence = bi.utilities.clean_string(i, allow=["#", "-"]).upper()
                assert len(sequence) > 0
                sequence_ok = True
            except:
                bi.log("error", "Please introduce a valid sequence:", i)
        bi.log(2, "Input sequence:\n", sequence)
        details["sequence"] = sequence

        pdb_ok = False
        while not pdb_ok:
            bi.log("header", "Please introduce PDB (code or path) to overlay prediction")
            bi.log(1, "Leave blank for no structural representation")
            i = input("\n>>> ")

            if len(i) == 0:
                bi.log(2, "No structure selected")
                details["pdb_code"] = None
                details["pdb_path"] = None

                pdb_ok = True
            elif len(i) == 4:
                details["pdb_code"] = i.upper()
                details["pdb_path"] = None
                bi.log(2, "PDB code:", details["pdb_code"])
                pdb_ok = True
            elif os.path.exists(i):
                details["pdb_path"] = os.path.abspath(i)
                details["pdb_code"] = None
                bi.log(2, "PDB path:", details["pdb_path"])
                pdb_ok = True
            else:
                bi.log("error", "Invalid PDB code (or file does not exist):", i)




    def predict(details):
        import torch
        # INPUT READY
        bi.log("header", "Input received!")
        print(json.dumps(details, indent=4))

        from models import get_model_class
        model_class = get_model_class(details["model_name"].split("_")[0].split("-")[0])
        bi.log(1, "Model class:", model_class)

        os.makedirs("pred", exist_ok=True)
        n = 0
        fname = f"prediction_{bi.utilities.strings.add_front_0(n, 3)}"
        while os.path.exists(f"pred/{fname}"):
            n += 1
            fname = f"prediction_{bi.utilities.strings.add_front_0(n, 3)}"

        bi.log("header", f"Starting prediction: {fname}")


        pred_folder = os.path.abspath(f"pred/{fname}")
        os.makedirs(pred_folder)
        bi.log(1, "Prediction folder:", pred_folder)

        structure = None
        if details["pdb_code"] is not None:
            download_folder = bi.biopython.downloadPDB("pred", fname, [details["pdb_code"]], file_format="cif")
            file_path = os.path.join(download_folder, f"{details['pdb_code']}.cif")
            structure = bi.biopython.loadPDB(file_path)
        elif details["pdb_path"] is not None:
            if os.path.exists(details["pdb_path"]):
                file_path = details["pdb_path"]
                structure = bi.biopython.loadPDB(details["pdb_path"])
            else:
                bi.log("error", "Provided path does not exist: {}".format(details["pdb_path"]))
                exit()
        print(structure)

        bi.log("header", "Sequence:")
        embedding_model = details["model_info"]["embeddings"]["model"]
        if embedding_model  == "SaProt":
            labels = {"A":{str(n):{
                "resn":name,
                "fs": "#",
                "label": None,
                "res":n,
            } for n, name in enumerate(sequence)}}
        else:
            raise Exception("Unknown model")

        print("".join([r["resn"] for r in labels["A"].values()]))
        print("length:", len(labels["A"]))

        with open(f"pred/{fname}/{fname}.fasta", "w") as f:
            f.write(f"\n>{fname}_query\n")
            f.write("".join([r["resn"] for r in labels["A"].values()]))

        json.dump(labels, open(f"pred/{fname}/{fname}.labels.json", "w"))

        #print(dssp)
        generate_embeddings(fname, structure, pred_folder, pred_folder, model=embedding_model, mode="seq", predict=True)



        device = config["general"]["device"]

        model_path = f"models/{details['model_name']}.pth"
        model = model_class(input_dim=details["model_info"]["embeddings"]["dimensions"],
                            num_classes=len(details["model_info"]["labels"]["classes"]))
        model.load_state_dict(torch.load(model_path, weights_only=False))

        model.eval()
        preds = []
        labels_eval = []


        embeddings = torch.load(f"pred/{fname}/{fname}_A.pt")[0][2:-2]
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
        print("".join([index_to_ss(p) for p in preds]))
        print("len:", len(preds))

        with open(f"pred/{fname}/{fname}.fasta", "a") as f:
            f.write(f"\n>{fname}_indexes\n")
            f.write("".join([str(p) for p in preds])+"\n")
            f.write(f"\n>{fname}_preds\n")
            f.write("".join([index_to_ss(p) for p in preds])+"\n")

        if structure is not None:
            for chain in structure.get_chains():
                print(chain)
                for res, p in zip(chain.get_residues(), preds[2:-2]):
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


    predict(details)
    bi.log("end", "PREDICT")









