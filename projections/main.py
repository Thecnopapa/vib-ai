import os, sys, json
import numpy as np


def import_bi():
    global bi
    global bioiain
    local_bi = "local-bi" in sys.argv
    try:
        if local_bi:
            raise ImportError("bioiain")
        import bioiain
        import bioiain as bi

    except:
        try:
            import importlib
            sys.path.append("/home/iain/projects/bioiain")
            import src.bioiain as bi
            bioiain = bi
        except:
            raise ImportError("bioiain")

import_bi()





def get_PCA(code):
    from sklearn.decomposition import PCA
    import plotly.graph_objs as go
    import matplotlib.pyplot as plt
    file_folder = bi.biopython.downloadPDB("./data", "test", pdb_list=[code], file_format="cif", overwrite=False)
    structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{code}.cif"))
    print(structure)
    chain = [c for c in structure[0].get_chains() if c.id =="A"][0]
    resids = [r for r in chain.get_residues()]
    coords = [a.coord for a in chain.get_atoms() if a.id == "CA"]
    print(coords[:10])
    print(len(coords))
    pca = PCA(n_components=3)
    pca.fit(coords)
    print(pca.components_)

    projected = pca.transform(coords)
    print(projected[0:10])

    print(bi)
    #from bioiain import visualisation
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    ax.scatter(projected[:, 0], projected[:, 1], c="black", marker=".")
    ax.set_aspect("equal")

    #ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax.axis('off')
    #fig.savefig(f'{code}.png', bbox_inches=extent)
    fig.savefig(f"{code}_projected.png")

    for i in range(len(projected)-1):
        ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000050")
    fig.savefig(f"{code}_connected.png")
    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    plt.show(block=False)

    import PIL
    img = PIL.Image.open(f"{code}_connected.png")

    import torchvision as tv
    tensor = tv.transforms.functional.pil_to_tensor(img)
    print(tensor)
    print(tensor.shape)

    plt.imshow(tensor[0].numpy())
    plt.savefig(f"{code}_tensor.png")




get_PCA("1M2Z")










