

import os, sys, subprocess, json

from setup import config, bi, bioiain











def generate_labels(name, structure=None):
    label_dict = {c.id: {} for c in structure.get_chains()}
    method = config["labels"]["selected"]["method"]
    if method== "dssp":
        bi.log(3, "Detected method: DSSP")
        try:
            bi.log(3, "DSSP command:", config["general"]["dssp"])
        except:
            bi.log("error", "DSSP command not found")
            exit()
        #print(json.dumps(config["labels"]["selected"], indent=4))
        run_dssp(name, label_dict,
                 "data/"+config["data"]["selected"]["folder_name"],
                 config["labels"]["selected"]["save_folder"],
                 config["labels"]["selected"]["raw_folder"],
                 config["labels"]["selected"]["abbreviation"])

    elif method == "sasa":
        bi.log(3, "Detected method: SASA")
        raise Exception("SASA not implemented")

    else:
        bi.log("error", "Label generator method not recognised:", config["labels"]["selected"]["method"] )






def run_dssp(name, label_dict, data_folder, save_folder, raw_folder, abbreviation):
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    cmd = [config["general"]["dssp"], "--output-format", "dssp", f"{data_folder}/{name}.cif",
           f"{raw_folder}/{name}.dssp"]
    bi.log(3, " ".join(cmd))
    subprocess.run(cmd)


    with open(f"{raw_folder}/{name}.dssp", "r") as f:
        start = False
        for line in f:
            if "#  " in line:
                start = True
                continue
            if not start:
                continue
            if "!" in line:
                continue
            res = line[5:11].strip()
            ch = line[11]
            resn = line[13].upper()
            ss = line[16].replace(" ", "#")
            if line[10] != " ":
                bi.log("warning", f"Disordered atom: \n {line}")
            label_dict[ch][res] = {"res": res, "resn": resn, abbreviation: ss}
    if not start:
        bi.log("error", "Error reading DSSP file")
        exit()

    with open(f"{save_folder}/{name}.labels.json", "w") as f:
        f.write(json.dumps(label_dict, indent=4))


