
import os, sys, subprocess


from Bio.PDB.DSSP import dssp_dict_from_pdb_file


sys.path.append("/home/iain/projects/bioiain")

import src.bioiain as bi
from src.bioiain.utilities.DSSP import DSSP


bi.log("start", "internship > main.py")




file_folder = bi.imports.downloadPDB("./data", "other", ["1M2Z"], file_format="cif", overwrite=False)

bi.log(1, "File folder:", file_folder)



structure = None
#structure = bi.imports.recover("1MOT")


bi.log(1, "Structure:", structure)


if structure is None:
    pass
structure = bi.imports.loadPDB(os.path.join(file_folder, "1M2Z.cif"))
structure.pass_down()
bi.log(1, "Structure:", structure)
bi.log("header", structure)

#structure.export(structure_format="cif")
model = structure[0]
print(model)
model.__getitem__ = model._getitem
model.export()
print(model.__getitem__)
print(model.get_list())
print(model.child_dict)
print(model[0])

# p = PDBParser()
# structure = p.get_structure("1MOT", "data/other/1MOT.cif")
# model = structure[0]
# print(DSSP)
# dssp_dict = dssp_dict_from_pdb_file("data/other/1MOT.pdb")
# print(dssp_dict)
os.makedirs("out", exist_ok=True)
os.makedirs("out/dssp", exist_ok=True)
dssp = DSSP(model, "out/dssp/1M2Z.cif", file_type="DSSP")
print(dssp)
print("residues:", len(dssp))


for key in dssp.keys():
    chain_id, res_id = key
    aa = dssp[key][1]          # Amino acid
    ss = dssp[key][2]          # Secondary structure
    #print(chain_id, res_id, aa, ss)

def run_foldseek(filename):
    "./SaProt/bin/foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 ./data/other/1M2Z.cif ./out/foldseek/1M2Z.tsv"

    cmd = ["./SaProt/bin/foldseek",
           "structureto3didescriptor", "-v", "0", "--threads", "1",
           "--chain-name-mode", "1", f"./data/other/{filename}.cif",
           f"./out/foldseek/{filename}.tsv"

    ]
    subprocess.run(cmd)

run_foldseek("1M2Z")





