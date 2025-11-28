

import os, sys, subprocess, json

from bioiain.biopython import ss_to_index

from setup import config, bi, bioiain

print(bioiain)
from bioiain.utilities import d3to1
from bioiain.visualisation import quick_display







def generate_labels(name, structure=None):
    label_dict = {c.id: {} for c in structure.get_chains() if len(c) >0}
    method = config["labels"]["selected"]["method"]
    if method== "dssp":
        bi.log(3, "Selected method: DSSP")
        try:
            bi.log(3, "DSSP command:", config["general"]["dssp"])
        except:
            bi.log("error", "DSSP command not found")
            exit()
        #print(json.dumps(config["labels"]["selected"], indent=4))
        return run_dssp(name, label_dict,
                 "data/"+config["data"]["selected"]["folder_name"],
                 config["labels"]["selected"]["save_folder"],
                 config["labels"]["selected"]["raw_folder"],
                 config["labels"]["selected"]["abbreviation"],
                        structure=structure)

    elif method == "sasa":
        bi.log(3, "Selected method: SASA")
        assert structure is not None
        calculate_sasa(name, label_dict, structure,
                       "data/" + config["data"]["selected"]["folder_name"],
                       config["labels"]["selected"]["save_folder"],
                       config["labels"]["selected"]["abbreviation"])

    else:
        bi.log("error", "Label generator method not recognised:", config["labels"]["selected"]["method"] )



def calculate_sasa(name, label_dict, structure, data_folder, save_folder, abbreviation):
    from Bio.PDB.SASA import ShrakeRupley
    os.makedirs(save_folder, exist_ok=True)
    probe_radius = config["labels"]["selected"]["probe-radius"]
    scope = config["labels"]["selected"]["scope"]
    threshold = config["labels"]["selected"]["threshold"]


    SR = ShrakeRupley(probe_radius=probe_radius)
    SR.compute(structure, level=scope)

    for chain in structure[0].get_chains():
        ch = chain.id
        for res in chain.get_residues():
            #print(ch, res.id, res.sasa)
            #print(res.id[0])
            if res.id[0] == " ":
                try:
                    label_dict[ch][len(label_dict[ch])] = {
                        "res": len(label_dict[ch]),
                        "resn3": res.resname,
                        "resn": d3to1[res.resname],
                        abbreviation: res.sasa,
                        "label": int(res.sasa > threshold),
                    }
                    ca = [a for a in res if a.id == "CA"][0]
                    #print("sasa:", res.sasa)
                    ca.bfactor = float(res.sasa > threshold)
                except:
                    bi.log("warning", "Residue not recognised:", res.resname)
    with open(f"{save_folder}/{name}.labels.json", "w") as f:
        f.write(json.dumps(label_dict, indent=4))




def run_dssp(name, label_dict, data_folder, save_folder, raw_folder, abbreviation, structure=None):
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)

    if os.path.exists(f"{save_folder}/{name}.labels.json") and not config["general"]["force"]:
        bi.log(3, "Label already exists")
        return True

    cmd = [config["general"]["dssp"], "--output-format", "mmcif", "-v", f"{data_folder}/{name}.cif",
           f"{raw_folder}/{name}.dssp"]
    bi.log(3, " ".join(cmd))
    subprocess.run(cmd)

    real_chains = list(label_dict.keys())
    halucination_pointer = {}
    #print("REAL", real_chains)
    dssp_dict = {}
    with open(f"{raw_folder}/{name}.dssp", "r") as f:
        start_sum = False
        start_bridge = False
        for line in f:
            if "_dssp_struct_summary" in line:
                start_sum = True
                continue
            if "_dssp_struct_bridge" in line:
                start_bridge = True
                continue


            if "#" in line:
                start_sum = False
                if start_bridge:
                    break
                continue
            if start_sum:
                #print(line)
                line = line.split(" ")
                line = [l for l in line if l != ""]
                #print(line)
                res = line[2]
                dch = line[1]
                if dch not in dssp_dict.keys():
                    dssp_dict[dch] = {}
                ch = real_chains[list(dssp_dict.keys()).index(dch)]

                #print("DSSP:", dch, "REAL:", ch)
                resn = line[3].upper()
                ss = line[4].replace(".", "#")

                if ch not in label_dict.keys():
                    continue
                try:
                    dssp_dict[dch][len(dssp_dict[dch])] = {"res": res, "resn": bi.utilities.d3to1[resn], "resn3":resn, abbreviation: ss, "label": ss_to_index(ss)}
                except:
                    dssp_dict[dch][len(dssp_dict[dch])] = {"res": res, "resn": None, "resn3":resn, abbreviation: ss, "label": ss_to_index(ss)}
            elif start_bridge:
                # print(line)
                line = line.split(" ")
                line = [l for l in line if l != ""]
                # print(line)
                dch = line[3].upper()
                rch = line[5].upper()

                if dch not in halucination_pointer.keys():
                    halucination_pointer[dch] = rch
    #print(halucination_pointer)
    [bi.log(4, "DSSP:", dch, "->>", "REAL:", rch) for dch, rch in halucination_pointer.items()]

    for dch, rch in halucination_pointer.items():
        label_dict[rch] = dssp_dict[dch]




    #print("DSSP", dssp_dict.keys())
    if not start_bridge:
        bi.log("error", "Error reading DSSP file for:", name)

    with open(f"{save_folder}/{name}.labels.json", "w") as f:
        f.write(json.dumps(label_dict, indent=4))


    return True


