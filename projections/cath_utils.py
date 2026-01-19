
import os
import sys


def pdbs_from_cath(similarity=None, cath_folder="cath", file_name="cath-domain-list.txt", save_to="data"):

    if similarity is not None:
        cath_file = f"{file_name.split('.')[0]}-S{similarity}.{file_name.split('.')[1]}"
    else:
        cath_file = f"{file_name}"

    cath_path = os.path.join(cath_folder, cath_file)
    save_path = os.path.join(save_to, cath_file)
    print(cath_path)
    assert os.path.exists(cath_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pdbs = []
    with open(cath_path) as f:
        for line in f:
            if line.startswith("#") or line == "\n":
                continue
            print(line, end="\r")
            pdbs.append(line[:4].upper())

    unique_pdbs = sorted((set(pdbs)))

    with open(save_path, "w") as f:
        f.write("\n".join(unique_pdbs))

    print(f"{len(unique_pdbs)} saved at: {save_path}")



if __name__ == "__main__":
    print(sys.argv)
    save_folder = "../internship/data"
    cath_file = "cath-dataset-nonredundant-S20.list"
    if len(sys.argv) > 1:
        pdbs_from_cath(similarity=sys.argv[1], file_name=cath_file, save_to=save_folder)
    else:
        pdbs_from_cath(save_to=save_folder, file_name=cath_file)


