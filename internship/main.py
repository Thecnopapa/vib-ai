
import os, sys


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
bi.log(1, "Structure:", structure)
bi.log("header", structure)

#structure.export(structure_format="cif")
model = structure[0]
print(model)
model.__getitem__ = model._getitem

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


exit()
model = structure[0]
print(model)
print(structure.paths["original"])
DSSP(model, structure.paths["original"] )

