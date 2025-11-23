
import os, sys, subprocess, json
import numpy as np
sys.path.append("/home/iain/projects/bioiain")

import src.bioiain as bi
from src.bioiain.utilities.DSSP import ss_to_index


from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def run_dssp(structure, filename, data_folder, out_folder="out/dssp"):
    os.makedirs("out", exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    cmd = ["dssp", "--output-format", "dssp", "--verbose", f"{data_folder}/{filename}.cif",
           f"{out_folder}/{filename}.dssp"]
    print(" ".join(cmd))
    subprocess.run(cmd)

    dssp_dict = {c.id: {} for c in structure.get_chains()}
    # print(dssp_dict)
    with open(f"{out_folder}/{filename}.dssp", "r") as f:
        start = False

        for line in f:

            if "#  " in line:
                start = True
                continue
            if not start:
                continue
            l = line.split(" ")
            l = [cl for cl in l if cl != ""]

            # res = l[1]
            # ch = l[2]
            # resn = l[3]
            # ss = l[4]
            if "!" in line:
                continue
            res = line[5:11].strip()
            ch = line[11]
            resn = line[13].upper()
            ss = line[16].replace(" ", "#")
            if line[10] != " ":
                bi.log("warning", f"Disordered atom: \n {line}")
                # exit()
                # continue

            # print(ch, res, resn, ss)

            dssp_dict[ch][res] = {"res": res, "resn": resn, "ss": ss}
    if not start:
        bi.log("error", "Error reading DSSP file")
        exit()

    return dssp_dict


def run_foldseek(structure, filename, dssp_dict, data_folder):
    bi.log(2, "running foldseek...")
    "./SaProt/bin/foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 ./data/other/1M2Z.cif ./out/foldseek/1M2Z.tsv"
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/foldseek", exist_ok=True)
    cmd = ["./SaProt/bin/foldseek",
           "structureto3didescriptor", "-v", "0", "--threads", "4",
           "--chain-name-mode", "1", f"{data_folder}/{filename}.cif",
           f"./out/foldseek/{filename}.csv"
           ]
    print(" ".join(cmd))
    subprocess.run(cmd)
    done_chains = []
    with open(f"./out/foldseek/{filename}.csv", "r", encoding="utf-8") as f:
        for line in f:
            ch = line.split("\t")[0].split("_")[1].split(" ")[0]
            if ch in done_chains:
                continue
            done_chains.append(ch)
            bi.log(3, "foldseek out:", ch)
            l = line.split("\t")[2]
            toks = [t for t in l]
            print(len(dssp_dict[ch].items()), len(toks))
            if not len(dssp_dict[ch].items()) == len(toks):
                dssp_dict.pop(ch)
                bi.log("warning", "foldseek tokens and dssp_dict do not match:", ch)
                continue
            for n, (k, v) in enumerate(dssp_dict[ch].items()):
                if toks[n] == " ":
                    toks = "-"
                dssp_dict[ch][k]["fs"] = toks[n]
    return dssp_dict


def generate_embeddings(dssp_dict, name, folder="out/SaProt", no3D=False):
    bi.log(2, "Generating embeddings...")
    seqs = {k: [] for k in dssp_dict.keys()}
    seqs3D = {k: [] for k in dssp_dict.keys()}
    #print(dssp_dict.items())
    for k, v in dssp_dict.items():
        #print(v)
        #print(k)

        if not no3D:
            try:
                seqs3D[k] = ["{}{}".format(i["resn"].upper(), i["fs"].lower()) for i in v.values()]
            except:
                continue
        seqs[k] = ["{}{}".format(i["resn"].upper(), "#") for i in v.values()]

        #print(seqs)
        #print(seqs3D)
        #print("######")

    for ch in seqs.keys():
        # Load model directly

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
        #print(inputs)

        with torch.no_grad():
            outputs = modelS(**inputs, output_hidden_states=True)
        # print(outputs)

        # outputs.hidden_states is a tuple of all layers, including embeddings
        # Shape of each layer: [batch_size, sequence_length, hidden_dim]
        all_hidden_states = outputs.hidden_states

        # Last layer hidden states
        last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
        #print(last_hidden.shape)  # ['<cls>', 'M#', 'E#', 'V#', 'Q#', '<eos>']
        #print(last_hidden)

        os.makedirs(f"{folder}/seq_only", exist_ok=True)
        torch.save(last_hidden, f"{folder}/seq_only/{name}_{ch}.pt")

        if not no3D:
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
            #print(last_hidden.shape)
            #print(last_hidden)
            os.makedirs(f"{folder}/full", exist_ok=True)
            torch.save(last_hidden, f"{folder}/full/{name}_{ch}.pt")
            json.dump(dssp_dict[ch], open(f"{folder}/full/{name}_{ch}.json", "w"))
    return os.path.abspath(folder)