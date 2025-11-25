

import os, sys, subprocess, json

from setup import config, bi, bioiain
from Bio.PDB import Polypeptide










def generate_labels(name, structure=None):
    label_dict = {c.id: {} for c in sorted(structure.get_chains())}
    method = config["labels"]["selected"]["method"]
    if method== "dssp":
        bi.log(3, "Selected method: DSSP")
        try:
            bi.log(3, "DSSP command:", config["general"]["dssp"])
        except:
            bi.log("error", "DSSP command not found")
            exit()
        assert structure is not None
        #print(json.dumps(config["labels"]["selected"], indent=4))
        return run_dssp(name, label_dict,
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

    if os.path.exists(f"{save_folder}/{name}.labels.json") and not config["general"]["force"]:
        bi.log(3, "Label already exists")
        return True

    cmd = [config["general"]["dssp"], "--output-format", "mmcif", f"{data_folder}/{name}.cif",
           f"{raw_folder}/{name}.dssp"]
    bi.log(3, " ".join(cmd))
    subprocess.run(cmd)


    with open(f"{raw_folder}/{name}.dssp", "r") as f:
        start = False
        for line in f:
            if "_dssp_struct_summary" in line:
                start = True
                continue

            if not start:
                continue
            if "#" in line:
                break

            #print(line)
            line = line.split(" ")
            line = [l for l in line if l != ""]
            #print(line)
            res = line[2]
            ch = line[1]
            resn = line[3].upper()
            ss = line[4].replace(".", "#")

            if ch not in label_dict.keys():
                continue

            label_dict[ch][len(label_dict[ch])] = {"res": res, "resn": bi.utilities.d3to1[resn], "resn3":resn, abbreviation: ss}

    if not start:
        bi.log("error", "Error reading DSSP file")
        exit()

    with open(f"{save_folder}/{name}.labels.json", "w") as f:
        f.write(json.dumps(label_dict, indent=4))

    return True


