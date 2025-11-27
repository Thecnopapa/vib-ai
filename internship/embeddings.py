
import os, sys, subprocess, json

from setup import config, bi, bioiain





from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch









def generate_embeddings(name, structure=None, label_path=None, save_folder=None, model=None, mode=None, predict=False):
    if model is None:
        model = config["embeddings"]["selected"]["model"]
    if model== "SaProt":
        bi.log(3, "Selected model: SaProt")
        if predict:
            assert mode is not None
            assert label_path is not None
            assert save_folder is not None
        else:
            mode = config["embeddings"]["selected"]["mode"]
            label_path = config["labels"]["selected"]["save_folder"]
            save_folder = config["embeddings"]["selected"]["save_folder"]
        bi.log(3, "Selected mode:", mode)

        foldseek_path = None
        if mode == "full":
            try:
                bi.log(3, "Foldseek command:", config["general"]["foldseek"])
            except:
                bi.log("error", "Foldseek command not found")
                exit()
            run_foldseek(name,
                         "data/"+config["data"]["selected"]["folder_name"],
                         config["embeddings"]["selected"]["foldseek_folder"],
                         label_path)


            foldseek_path = config["embeddings"]["selected"]["foldseek_folder"]


        run_saprot(name, mode, foldseek_path, label_path, save_folder)


    else:
        bi.log("error", "Embedding generator model not recognised:", config["embeddings"]["selected"]["model"] )





def run_foldseek(filename, data_folder, raw_folder, label_folder):
    os.makedirs(raw_folder, exist_ok=True)

    if os.path.exists(f"{raw_folder}/{filename}.foldseek.json") and not config["general"]["force"]:
        bi.log(3, "Foldseek already calculated")
        return True
    label_dict = json.load(open(f"{label_folder}/{filename}.labels.json"))
    cmd = [config["general"]["foldseek"],
           "structureto3didescriptor", "-v", "0", "--threads", "4",
           "--chain-name-mode", "0", f"{data_folder}/{filename}.cif",
           f"{raw_folder}/{filename}.foldseek.csv"
           ]
    bi.log(4," ".join(cmd))
    subprocess.run(cmd)
    done_chains = []
    foldseek_dict = {k:None for k in label_dict.keys()}
    with open(f"{raw_folder}/{filename}.foldseek.csv", "r", encoding="utf-8") as f:

        for line , ch in zip(f, foldseek_dict.keys()):

            if ch in done_chains:
                continue
            done_chains.append(ch)
            #print(ch)


            rns, tks = line.split("\t")[1:3]
            resns = [r for r in rns]
            toks = [t for t in tks]

            bi.log(3, "foldseek out:", ch, len(resns), len(toks))


            if not len(resns) == len(toks):
                print(resns)
                print(toks)
                print(len(resns), len(toks))
                foldseek_dict.pop(ch)
                bi.log("warning", "foldseek tokens and dssp_dict do not match:", ch)
                exit()
            foldseek_dict[ch] = {}
            for n, (r, t) in enumerate(zip(resns, toks)):
                if toks[n] == " ":
                    toks = "-"
                foldseek_dict[ch][n] = {"fs": t, "resn": r}
    json.dump(foldseek_dict, open(f"{raw_folder}/{filename}.foldseek.json", "w"), indent=4)
    return True



def run_saprot(name, mode, foldseek_path, label_path, save_folder):
    bi.log(3, "Running SaProt, mode:", mode)
    label_dict = json.load(open(f"{label_path}/{name}.labels.json"))
    #print(label_dict.keys())
    fs_keys = label_dict.keys()
    if mode == "full":
        assert foldseek_path is not None
        foldsek_dict = json.load(open(f"{foldseek_path}/{name}.foldseek.json"))
        #print(foldsek_dict.keys())
        assert label_dict.keys() == foldsek_dict.keys()
        fs_keys = foldsek_dict.keys()

    seqs = {}
    #print(label_dict.keys(), foldsek_dict.keys())

    for ch, fch in zip(label_dict.keys(), fs_keys):
        bi.log(4, "Merging foldseek_dict:", ch, fch)
        if mode == "full":
            if foldsek_dict[ch] is None:
                bi.log("warning", f"chain {ch} has no foldseek data")
                continue
            #print(foldsek_dict[ch])


            if len(label_dict[ch]) != len(foldsek_dict[ch]):

                bi.log("warning", "label and foldseek_dict do not match:", ch, len(label_dict[ch]), len(foldsek_dict[ch]))
                continue
            try:
                seqs[ch] = [f"{l["resn"].upper()}{f["fs"].lower()}" for l,f in zip(label_dict[ch].values(),foldsek_dict[ch].values())]
            except:
                bi.log("warning", "unknown atom in chain:", ch)
                [bi.log("warning", f"{r["res"]} -> {r["resn"]} / {r["resn3"]}") for r in label_dict[ch].values() if None in [r["res"], r["resn"], r["resn3"]]]
        elif mode == "seq":
            seqs[ch] = [f"{l["resn"]}#" for l in label_dict[ch].values()]
        else:
            bi.log("error", "Unknown SaProt mode:", mode)
    #print("FOLDSEEK", seqs.keys())


    for ch in seqs.keys():
        # Load model directly
        bi.log(4, "Generating embeddings:", ch)
        device = config["general"]["device"]
        if mode == "full":
            tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_35M_AF2")
            model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_35M_AF2")
        elif mode == "seq":
            tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_35M_AF2_seqOnly")
            model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_35M_AF2_seqOnly")
        else:
            bi.log("error", "Unknown SaProt mode:", mode)


        model.eval()
        model.to(device)

        seq = "".join(seqs[ch])

        inputs = tokenizer(seq, return_tensors="pt").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        #print(inputs)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # print(outputs)

        # outputs.hidden_states is a tuple of all layers, including embeddings
        # Shape of each layer: [batch_size, sequence_length, hidden_dim]
        all_hidden_states = outputs.hidden_states

        # Last layer hidden states
        last_hidden = all_hidden_states[-1]  # [1, seq_len, hidden_dim]
        #print(last_hidden.shape)  # ['<cls>', 'M#', 'E#', 'V#', 'Q#', '<eos>']
        #print(last_hidden)

        os.makedirs(save_folder, exist_ok=True)
        torch.save(last_hidden, f"{save_folder}/{name}_{ch}.pt")

    return True


