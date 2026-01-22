import os, sys, json

from setup import config, bi, bioiain
from bioiain import log
from bioiain.biopython import downloadPDB, recover, loadPDB
from bioiain.symmetries.oligomer import Oligomer

print(bioiain)



def generate_dimers(code, file_folder):


    structure = recover(code)

    if structure is None:
        structure = loadPDB(os.path.join(file_folder, f"{code}.cif"))

    log("header", structure)

    structure.init_all()

    crystal = structure.get_crystals()

    crystal.set_params(
        min_monomer_length=50,
        oligomer_levels=[2],
        min_contacts=10,
        contact_threshold=15,
    )

    crystal.process(force="force" in sys.argv)

    print(crystal.paths["oligo_folder"])
    for file in os.listdir(crystal.paths["oligo_folder"]):
        if file.endswith(".data.json"):
            from bioiain.symmetries import Oligomer
            oligo = Oligomer.recover(file)
            print(oligo)
    return ["future labs"]



