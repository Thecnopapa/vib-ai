import os, sys, json, glob
import warnings
import numpy as np
import pandas as pd
from bioiain.utilities import str_to_list_with_literals, find_com
import PIL.Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from mpl_toolkits.axes_grid1 import ImageGrid
from parallel import *

sys.path.append('..')
np.random.seed(6)

files_selected = list()
from collections import defaultdict

file_folder = "/media/mari/Data/vib_leuven/datasets/cc_sasa/biomols"
files_list = os.listdir(file_folder)
data_csv = pd.read_csv(sys.argv[-1], sep="\t", header=0)
files_data = defaultdict(dict)
for i in data_csv.index:
    pdb = data_csv.at[i, "pdb"]
    biomol = data_csv.at[i, "biomol"]
    orient = data_csv.at[i, "orient"]
    oligo = data_csv.at[i, "oligo"]
    files_data[f"{pdb.upper()}_{biomol}.pdb"]["label"] = f"{oligo}_{orient}"
    files_data[f"{pdb.upper()}_{biomol}.pdb"]["chains"] = data_csv.at[i, "CC_chains"]
for file in files_list:
    if file in files_data.keys():
        files_selected.append(file)

def get_family_desc(fam, cath_folder="cath"):
    fam_list_names_file = os.path.join(cath_folder, "cath-superfamily-list.txt")
    with open(fam_list_names_file) as ff:
        for line in ff:
            if line[0] == "#":
                continue
            line = str_to_list_with_literals(line.expandtabs())
            #print(line)
            if line[0] != fam:
                continue
            return " ".join(line[3:])

def parse_cath(code, chain, domain=None, cath_folder="cath"):

    dom_list_file = os.path.join(cath_folder, "cath-domain-list.txt")




    with open(dom_list_file) as f:
        for line in f:
            if line[0] == "#":
                continue
            if line[:4].upper() != code.upper():
                continue

            comps = str_to_list_with_literals(line)
            c = comps[0][:4]
            ch = comps[0][4:5]
            dom = comps[0][5:7]

            if code.upper() != c.upper():
                continue

            if chain.upper() != ch.upper():
                continue

            if domain != None:
                if int(domain) != int(dom):
                    continue

            info = dict(
                dom_name = comps[0],
                class_number = comps[1],
                arch_number = comps[2],
                top_number = comps[3],
                hom_fam_number = comps[4],
                s35 = comps[5],
                s60 = comps[6],
                s95 = comps[7],
                s100 = comps[8],
                s100_count = comps[9],
                dom_len = comps[10],
                res = comps[11]
            )
            superfamily = f"{info['class_number']}.{info['arch_number']}.{info['top_number']}.{info['hom_fam_number']}"
            info["superfamily"] = superfamily
            description = get_family_desc(superfamily)
            info["description"] = description
            #print(json.dumps(info))
            return info
        return None



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




def cath_to_label(name, structure, label_folder="labels", force=False ):
    pass




def get_PCA(force=False, labs=True, images=True):
    if labs and images:
        print("\033]0;Generating labels and embeddings\a")
    elif labs:
        print("\033]0;Generating labels\a")
    elif images:
        print("\033]0;Generating embeddings\a")


    from sklearn.decomposition import PCA, SparsePCA
    sys.path.append(".")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    batches = split_iterable(sorted(files_selected))
    #[print(b) for b in batches]
    print("FILE_FOLDER:", file_folder)
    os.makedirs("labels", exist_ok=True)
    def generate_cath_labels(batch, do_labs, do_images):
        print("STARTING BATCH")
        for file in sorted(batch):
            code = file.split(".")[0]
            chaincounts = 0
            structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{file}"))
            labels = {}
            chains = list(structure.get_chains())
            label_path = f"labels/{code}.labels.json"
            #print(structure, end=":\t")
            print(structure)
            if ((not os.path.exists(label_path)) or force) and do_labs:
                for chain in chains:
                    if chain.id in files_data[file]["chains"]:
                        l = files_data[file]["label"]
                        labels[chaincounts] = {
                            "chain_id": chain.id,
                            "label": l
                        }
                        chaincounts+=1
                    #print(f"{chain.id}:{l}", end="\t")
                json.dump(labels, open(label_path, "w"), indent=4)
            #print("")

            if not do_images:
                continue

            os.makedirs("imgs", exist_ok=True)
            #if code not in ["1M2Z", "1P93"]:
            #    continue
            projected_path = f"imgs/projected/{code}.png"
            connected_path = f"imgs/connected/{code}.png"
            double_path = f"imgs/double/{code}.png"
            double_connected_path = f"imgs/double_connected/{code}.png"
            paths = (projected_path, connected_path, double_path, double_connected_path)

            projected = list()
            coordsCA = list()
            chainsRes = list()
            if any([not os.path.exists(p) for p in paths]) or force:
                for chain in structure[0]:
                    if chain.id not in files_data[file]["chains"]:
                        continue
                    for residue in chain:
                        for atom in residue:
                            coordsCA.append(atom.coord)
                            chainsRes.append(chain.id)
                    coordsCB = [a.coord for a in chain.get_atoms() if a.id == "CB"]
                    if len(coordsCA) < 5:
                        continue
                    for p in paths:
                        os.makedirs(os.path.dirname(p), exist_ok=True)

                pca = PCA(n_components=3, random_state=6)
                pca.fit(coordsCA)
                projected_chain = pca.transform(coordsCA) # Already centered in centroid of protein

                print(projected_chain)
                # distances from centroid (coords already centered)
                """
                distances = np.linalg.norm(projected_chain, axis=1)
                # find residue closest to centroid
                center_idx = np.argmin(distances)
                window = 30
                start = max(0, center_idx - window)
                end = min(len(projected_chain), center_idx + window)
                projected_chain = projected_chain[start:end]
                #projected = np.array([p for p in projected if p50Y+stdY >= p[1] >= p50Y-stdY])
                #projected = [p for p in projected if p > p50]
                
                for ele, c in zip(projected_chain, chainsRes):
                    projected.append(ele)
                """
                # Single Projected
                projected = np.array([p for p in projected_chain if -6 <= p[0] <= 6])

                fig = plt.figure(figsize=(1.28,1.28))
                ax = fig.add_subplot(111)
                ax.set_aspect("equal")
                ax.axis("off")

                ax.scatter(projected[:, 1], projected[:, 2], c="#00000050", marker=".")

                fig.savefig(projected_path, transparent=True)

                # Single connected

                for i in range(len(projected)-1):
                    ax.plot(projected[i:i+2, 1], projected[i:i+2, 2], color="#00000050")
                fig.savefig(connected_path, transparent=True)

                plt.close(fig)



    pool = ThreadPool()
    for b in batches:
        print(len(b), b[0], b[-1])
        pool.add(generate_cath_labels, b, do_labs=labs, do_images=images)
    pool.start(wait=True)




def image_classifier(mode="connected", train = True, decode=False, view=False, temp=False):
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from models import Net as M
    import torch
    import torch.nn as nn

    from models import SmallNetQMINST as SmallNet
    M = SmallNet

    dataset_name = "tetramers"
    file_folder = "/media/mari/Data/vib_leuven/datasets/cc_sasa/biomols"

    if "auto" in sys.argv:
        net_mode = "auto"
    else:
        net_mode = "normal"

    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Resize((64,64)),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    if train:
        print("\033]0;Training (curating)\a")


        structure_list = []
        labs = []

        print("DATASET:", file_folder)
        if file_folder is not None:
            print(f"Filtering {len(os.listdir(file_folder))} PDBs...")
            valid = 0
            no_label = 0
            no_image = 0
            global files_selected
            print("heheheh",files_selected)
            for n, file in enumerate(files_selected):
                code = file.split(".")[0]
                l_path = os.path.join("labels", f"{code}.labels.json")
                if os.path.exists(l_path):
                    lab_data = json.load(open(l_path))
                    labs.extend([v["label"] for v in lab_data.values()] )
                    structure_list.append(code)
                else:
                    print(n, file, "label not found", end="\r")
                    no_label += 1
                    continue
                if False:
                    i_path = os.path.join(f"imgs/{mode}", f"{code}*.png")
                    if glob.glob(i_path):
                        pass
                    else:
                        print(n, file, "images not found", end="\r")
                        no_image += 1
                        continue
                structure_list.append(code)
                valid +=1
            print()

            print("NO LABEL:", no_label)
            #print("NO IMAGE:", no_image)
            print("INVALID:", no_image+no_label)
            print("VALID:", valid)




            n_labs = {k:v for k,v in sorted([(l, labs.count(l)) for l in set(labs)], key= lambda x: x[1], reverse=True)}

            index_to_label = {}
            label_to_index = {}
            for k, v in n_labs.items():
                n = len(index_to_label)
                label_to_index[str(k)] = int(n)
                index_to_label[int(n)] = str(k)

            labs = index_to_label.values()
            all_labs = label_to_index.keys()

            print()
            print("N chains:", len(structure_list))
            print("N labs (n>10):", len(labs))
            print("N labs (all):", len(all_labs))
            # print(json.dumps({k:v for k,v in n_labs.items() if v>=10}, indent=4))
            img_folder = f"imgs/{mode}"

        import torchvision


        class QMNISTDataset(torchvision.datasets.QMNIST):
            def __getitem__(self, i):

                item, lab = super().__getitem__(i)
                item = torchvision.transforms.functional.pil_to_tensor(item)
                item = item / 256

                return item, lab



        class ImageDataset(Dataset):
            def __init__(self,   pdb_codes=None, img_folder=None, label_folder=None, init=True):
                if init:
                    print(f"ImageDataset: Loading data ({len(pdb_codes)} ids)")

                    assert img_folder is not None
                    self.detect_chain_mode(pdb_codes)


                else:
                    print(f"ImageDataset: Not iniliatised yet")

                self.img_folder = img_folder

                if label_folder is None:
                    self.label_folder = self.img_folder
                else:
                    self.label_folder = label_folder

                self.images = []
                self.labels = []
                self.codes = []
                self.chains = []
                self.image_dims = None

                if init:
                    self.init(pdb_codes)

            def detect_chain_mode(self, codes):
                codes = list(codes)
                print(codes)
                if len(codes[0].split("_")) == 2:
                    self.has_chains = False
                    self.has_separated_chains = False
                elif len(codes[0].split("_")) == 3:
                    self.has_chains = True
                    self.has_separated_chains = True
                else:
                    raise Exception(f"ImageDataset: provided list is not made of pdb codes and/or chains. Provided: {codes[0]}")
            def split(self, pdb_codes, target=0.2):
                self.detect_chain_mode(pdb_codes)
                return self.validate_input(pdb_codes, split=True, target=target)

            def init(self, pdb_codes):
                self.validate_input(pdb_codes)

                example_img = Image.open(os.path.join(self.img_folder, self.images[0]))
                # print(img)
                self.channels = 1
                self.image_dims = example_img.size[0]

                print(
                    f"ImageDataset: loaded {len(self)} images from {len(self.codes)} pdbs and {len(self.chains)} chains!")


            def validate_input(self, codes=None, split=False, target=0.2):
                print("codes",codes)
                if split:
                    splitted = {"test":{}, "train":{}}

                no_label = 0
                valid_chains = []
                valid_codes = []
                for file in os.listdir(self.img_folder):
                    name = file.split(".")[0]
                    code, biomol = name.split("_")
                    code = f"{code}_{biomol}"
                    if codes is not None:
                        if self.has_chains:
                            if not (f"{code}_{chain}" in codes):

                                continue
                        else:
                            if not (code in codes):
                                continue


                    l_path = os.path.join(self.label_folder, f"{code}.labels.json")
                    if os.path.exists(l_path):
                        try:
                            print(json.load(open(l_path)))
                            lab = json.load(open(l_path))["0"]["label"]
                            self.labels.append(lab)
                            self.images.append(file)
                        except KeyError:
                            no_label += 1
                            raise
                            continue
                    else:
                        no_label += 1
                        continue
                    valid_chains.append(f"{code}")
                    valid_codes.append(code)
                    if split:
                        #print(lab,lab in splitted["train"].keys() )
                        if lab in splitted["train"].keys():
                            ratio = len(splitted["test"][lab])/len(splitted["train"][lab])
                            if ratio > target:
                                splitted["train"][lab].append(f"{code}")
                            else:
                                splitted["test"][lab].append(f"{code}")
                        else:
                            splitted["train"][lab] = [f"{code}"]
                            splitted["test"][lab] = []
                            #print("new lab:", lab)
                if split:
                    for lab in sorted(splitted["train"].keys(), key= lambda x: len(splitted["train"][x]), reverse=True)[:10]:
                        print("-",lab, len(splitted["train"][lab]), len(splitted["test"][lab]))


                if no_label > 0:
                    print(f"ImageDataset: (Warning) Some labels were not found: {no_label}")

                self.codes = sorted(set(valid_codes))
                self.chains = sorted(set(valid_chains))
                if split:
                    tr = []
                    te = []
                    for trv, tev in zip(splitted["train"].values(), splitted["test"].values()):
                        #print(trv, tev)
                        for c in trv:
                            tr.append(c)
                        for c in tev:
                            te.append(c)
                    #print(json.dumps(splitted["train"],indent=4))
                    #print(json.dumps(splitted["test"],indent=4))
                    return tr, te

                return self.chains

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx, as_image=False):
                fname = self.images[idx]
                name = os.path.basename(fname).split(".")[0]
                code, biomol = name.split("_")
                code = f"{code}_{biomol}"

                i_path = os.path.join(self.img_folder, fname)
                l_path = os.path.join(self.label_folder, f"{code}.labels.json")
                keys = label_to_index.keys()
                label = label_to_index[json.load(open(l_path))["0"]["label"].lower().strip()]


                image = Image.open(i_path)
                #image = image.convert("RGB")
                if as_image:
                    return image, label
                image = transform(image)#[-1]#.resize((1,self.image_dims,self.image_dims))
                image = image[-1]
                #image = torch.sigmoid(image)

                # print("emb:", emb.shape, "lab:", lab)

                return image, label





        batch_size = 1





        if "QMNIST" in sys.argv:
            trainset = QMNISTDataset("datasets", download=True, what="train")
            testset =QMNISTDataset("datasets", download=True, what="test10k")
            trainset.channels = 1
            trainset.image_dims = 28
        else:
            assert os.path.exists(img_folder)
            print("IMG_FOLDER:", img_folder)
            #structure_list = list(ImageDataset(structure_list, img_folder=img_folder, label_folder="labels").chains)
            #print(len(structure_list))
            #print(structure_list[0:20])
            #train_list, test_list = train_test_split(structure_list, train_size=0.8, test_size=0.2, random_state=42)
            print("aqui",structure_list)
            print(files_selected)
            train_list, test_list = ImageDataset(img_folder=img_folder, label_folder="labels", init=False).split(structure_list)
            print(len(train_list), len(test_list))
            trainset = ImageDataset(train_list, img_folder=img_folder, label_folder="labels")
            testset = ImageDataset(test_list, img_folder=img_folder, label_folder="labels")



        print(f"ACTUAL VALID DATA [(img+label)/chain]: \033[0;36m{len(trainset)+len(testset)}\033[0m test/train: {len(testset)}/{len(trainset)} ({len(testset)/len(trainset):.2f})")

        # from models import test_layer
        # import matplotlib.pyplot as plt
        # plt, ax = plt.subplots(1,2)
        # i = trainset[0][0]
        # print(i)
        # ax[0].imshow(i[0])
        # t= test_layer(i).detach()
        # ax[1].imshow(t[0][0])
        #
        # plt.show()
        # input("Press Enter to continue...")
        # exit()


        from parallel import cpu_count
        collate_fn = None
        if "QMNIST" in sys.argv:

            labs = sorted(list(set([str(x[1]) for x in trainset])))
            label_to_index = {str(l):int(l) for l in labs}
            index_to_label = {int(l):str(l) for l in labs}
            print(label_to_index)
            print(index_to_label)
            n_labs = {str(k):0 for k in labs}
            print(labs)
            for _, l in trainset:
                n_labs[str(l)] += 1
            print(labs)
            print(n_labs)


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=0)






        net = M(len(labs), n_channels=trainset.channels, fig_size=trainset.image_dims, mode=net_mode)
        data = {}
        data["model_name"] = f"{net.__class__.__name__}_{mode}_{dataset_name}"
        data["name"] = str(net.__class__.__name__)
        data["mode"] = str(mode)
        data["dataset"] = str(dataset_name)
        data["label_to_index"] = dict(label_to_index)
        data["index_to_label"] = dict(index_to_label)
        data["n_features"] = len(labs)
        data["n_channels"]=int(trainset.channels)
        data["fig_size"]=int(trainset.image_dims)
        with open(data["model_name"]+".model.data.json", "w") as f:
            json.dump(data, f, indent=4)

        import torch.optim as optim
        import torch
        import torchvision

        from models import DiceLoss, SimpleLoss, SimpleImageLoss, RotLoss


        #criterion = SimpleLoss
        criterion = nn.MSELoss()

        optimizer = optim.Adam(net.f_net.parameters(), lr=0.0005)

        #i_criterion = DiceLoss
        i_loss_fn = nn.MSELoss()
        i_criterion = RotLoss
        i_optimizer = optim.Adam(net.r_net.parameters(), lr=0.0005)

        auto_optimiser = optim.Adam(net.auto_f_net.parameters(), lr=0.0001)


        epochs = 100
        splitsize = len(trainloader) // 10
        print(f"SPLITSIZE: {splitsize}")

        from torch.utils.tensorboard import SummaryWriter
        import datetime
        writer = SummaryWriter(log_dir=f"runs/{dataset_name}/{optimizer.__class__.__name__}-{i_optimizer.__class__.__name__}-{datetime.datetime.now()}")

        n_samples = 20

        if "QMNIST" in sys.argv:
            images = np.array([trainset[x][0].numpy() for x in range(n_samples)])
        else:
            images = np.array([[trainset.__getitem__(x, as_image=False)[0].numpy()] for x in range(n_samples)])
        #print(images)

        writer.add_graph(net, torch.Tensor(images[0]))
        labels = [trainset[x][1] for x in range(n_samples)]

        for n, (lab, img) in enumerate(zip(labels, images)):
            #print(lab, img.shape)
            writer.add_image(f"input/train ({lab}:{index_to_label[lab]})", torch.Tensor(img), 0)




        try:
            for epoch in range(epochs):  # loop over the dataset multiple times
                print(f"\033]0;Training (E={epoch+1}/{epochs})\a")

                torch.save(net.state_dict(), "./model.temp.pth")
                with open("./model.temp.data.json", "w") as f:
                    json.dump(data, f, indent=4)


                running_loss = 0.0
                running_i_loss = 0.0
                running_auto_loss = 0.0
                if os.environ.get("SLURM_CPUS_PER_TASK", None) is None:
                    print(f'{0:4d}[{epoch + 1:2d}, {0:5d}] loss: {0.:5.3f} i-loss: {0.:5.3f}',end="\r")

                for i, d in enumerate(trainloader):
                    step = epoch*10+i//splitsize

                    imgs, labels = d

                    if len(imgs.shape) == 3:
                        imgs = imgs.reshape([1, *imgs.shape])

                    truth = [0.] * len(labs)
                    truth[int(labels[0])] = 1.
                    truth = torch.Tensor(truth)
                    truth = truth.reshape(1, *truth.shape)

                    if net_mode == "auto":
                        auto_optimiser.zero_grad()
                        #print("\nlab:", int(labels[0]))
                        #print("\nlabname:", index_to_label[int(labels[0])])

                        auto_loss = i_criterion(net.forward(imgs), imgs, i_loss_fn, label=index_to_label[int(labels[0])]) + criterion(net.forward(imgs, mode="normal"), truth)
                        auto_loss.backward()
                        running_auto_loss += auto_loss.item()
                        auto_optimiser.step()
                    else:
                        optimizer.zero_grad()
                        i_optimizer.zero_grad()

                        outputs = net(imgs)
                        # pred = torch.max(outputs, 1).indices[0]
                        # print(pred)


                        loss = criterion(outputs, truth)
                        loss.backward(retain_graph=False)
                        running_loss += loss.item()

                        i_outputs = net.backward(truth)
                        i_loss = i_criterion(i_outputs, imgs, i_loss_fn)
                        i_loss.backward()

                        running_i_loss += i_loss.item()

                        optimizer.step()
                        i_optimizer.step()



                    if os.environ.get("SLURM_CPUS_PER_TASK", None) is None:
                        print(
                            f'{step+1:4d}[{epoch + 1:2d}, {i%(splitsize+1):5d}] loss: {running_loss / (i % splitsize + 1):5.3f} i-loss: {running_i_loss / (i % splitsize+1):5.3f} auto-loss: {running_auto_loss / (i % splitsize+1):5.3f}', end="\r")

                    if i % splitsize == 0 and i != 0:  # print every 1000 mini-batches
                        print(f'{step:4d}[{epoch + 1:2d}, {i:5d}] loss: {running_loss / splitsize:5.3f} i-loss: {running_i_loss / splitsize:5.3f} auto-loss: {running_auto_loss / splitsize:5.3f}', end = "\n")
                        writer.add_scalar(f"loss/encode ({criterion.__class__.__name__})", running_loss / splitsize, step)
                        writer.add_scalar(f"loss/decode ({i_criterion.__class__.__name__})", running_i_loss / splitsize, step)
                        writer.add_scalar(f"loss/auto ({i_criterion.__class__.__name__})", running_auto_loss / splitsize, step)

                        running_loss = 0.0
                        running_i_loss = 0.0
                        running_auto_loss = 0.0

                        for n, (f, r) in enumerate(zip(net.f_layers[:-1], net.r_layers[:-1][::-1])):
                            #print(layer.__dict__)
                            #print(n, f, r)

                            if hasattr(r, "weight"):
                                writer.add_histogram(f"weight/decoding/{n}/{r.__class__.__name__}", r.weight, epoch * 10 + i // splitsize)
                            if hasattr(r, "bias"):
                                writer.add_histogram(f"bias/decoding/{n}/{r.__class__.__name__}", r.bias, epoch * 10 + i // splitsize)
                            if hasattr(f, "weight"):
                                writer.add_histogram(f"weight/encoding/{n}/{f.__class__.__name__}", f.weight, epoch * 10 + i // splitsize)
                            if hasattr(f, "bias"):
                                writer.add_histogram(f"bias/encoding/{n}/{f.__class__.__name__}", f.bias, epoch * 10 + i // splitsize)


                        with open("./model.temp.data.json", "w") as f:
                            json.dump(data, f, indent=4)

                        with torch.no_grad():
                            #print(labs)
                            for l in labs:
                                try:
                                    bits = [0]*len(labs)
                                    #print(bits)
                                    if "QMNIST" in sys.argv:
                                        n = int(l)
                                    else:
                                        n = label_to_index[l]
                                    bits[n] = 1
                                    #print(bits)
                                    bits = torch.Tensor(bits)
                                    #print("BITS LAB:", bits)
                                    dec = net.backward(bits, mode="normal")
                                    #print(dec)
                                    #print(dec.shape)
                                    writer.add_image(f"classes/{l}", dec, step)
                                except Exception as e:
                                    print(e)
                                    raise e
                                    continue
                PATH = f'./{net.__class__.__name__}_{mode}_{dataset_name}.model.pth'
                torch.save(net.state_dict(), PATH)
            print()
            print('Finished Training ({} epochs)'.format(epochs))

            PATH = f'./{net.__class__.__name__}_{mode}_{dataset_name}.model.pth'
            torch.save(net.state_dict(), PATH)

        except KeyboardInterrupt:
            PATH = f'./model.temp.pth'

        print(f"\033]0;Training (postprocess)\a")
        print("POSTRPROCESSING...")

        #dataiter = iter(testloader)
        #images, labels = next(dataiter)

        try:
            net = M(n_features = len(labs), n_channels=trainset.channels, fig_size=trainset.image_dims)
            net.load_state_dict(torch.load(PATH, weights_only=True))
        except:
            print("Temp model does not match!")
            exit()

        #outputs = net(images)

        #_, predicted = torch.max(outputs, 1)

        correct = 0
        total = 0
        pres = []
        truths = []

        correct_pred = {classname: 0 for classname in sorted(labs)}
        total_pred = {classname: 0 for classname in sorted(labs)}
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:

                images, labels = data
                predicts = net(images, mode="normal")
                #print(imgs.shape, labels)
                for img, label, pred in zip(images, labels, predicts):
                    #print(img.shape, label, pred)


                    pred, pred_index = torch.max(pred, 0)
                    pred_index = int(pred_index)
                    pred_label = index_to_label[pred_index]
                    label = index_to_label[int(label)]
                    label_index = label_to_index[label]
                    #print("label:", label, "index:", label_index)
                    #print(f"pred: {float(pred):.3f} index: {pred_index} label: {pred_label}")
                    total += 1
                    total_pred[label] += 1
                    pres.append(pred_label)
                    truths.append(label)
                    #print(pres[-1], truths[-1])
                    #print("correct: ", label_index == pred_index)


                    if label_index == pred_index:
                        correct_pred[label] += 1
                        correct += 1





        print(f'Accuracy of {len(testset)}/{len(trainset)} test images: {100 * correct /total:.1f} %')


        from internship.plotting import plot_confusion
        plot_confusion(truths, pres, f"{net.__class__.__name__}_{mode}_{dataset_name}", 100 * correct / total, sorted(labs))



        columns = [
            "accuracy",
            "correct",
            "total",
            "n_samples",
        ]



        os.makedirs("dataframes", exist_ok=True)
        df_path = f"dataframes/results.csv"
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            for col in columns:
                if col not in df.columns:
                    df.insert(len(df.columns), col, [None]*len(df))
        else:
            columns = ["title", "label",  "dataset", "mode"] + columns
            df = pd.DataFrame(columns=columns)


        df.set_index(["label", "dataset", "mode"], inplace=True, drop=False)
        df.sort_index(level="label", inplace=True)

        #print(df)
        #print("LEXSORTED:", df.index.is_monotonic_increasing, )


        for classname, correct_count in sorted([(k,v) for k, v in correct_pred.items() if k in labs], key=lambda x: n_labs[x[0]], reverse=True):
            #print(total_pred)
            df.sort_index(level="label", inplace=True)
            title = get_family_desc(classname)
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

            df.loc[(classname, dataset_name, mode), "label"] = classname
            df.loc[(classname, dataset_name, mode), "title"] = title
            df.loc[(classname, dataset_name, mode), "dataset"] = dataset_name
            df.loc[(classname, dataset_name, mode), "mode"] = mode

            df.loc[(classname, dataset_name, mode), "n_samples"] = n_labs[classname]

            if total_pred[classname] == 0:
                accuracy = None
                print(f'Accuracy for class ({label_to_index[classname]:3d}) -> {str(classname):12s}: \t{accuracy}\t(not in test set)\tin data: {n_labs[classname]:3d}\ttitle: {title}')

            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class ({label_to_index[classname]:3d}) -> {str(classname):12s}: \t{accuracy:4.2f}%\tcorrect:  {correct_count:3d}/{total_pred[classname]:3d}\tin data: {n_labs[classname]:3d}\ttitle: {title}')

                df.loc[(classname, dataset_name, mode), "correct"] = correct_count
                df.loc[(classname, dataset_name, mode), "total"] = total_pred[classname]

            df.loc[(classname, dataset_name, mode), "accuracy"] = accuracy
            warnings.simplefilter(action='default', category=pd.errors.PerformanceWarning)
            df.sort_index(level="label", inplace=True)

        df.sort_index(level="label", inplace=True)
        df.to_csv(df_path, index=False)
        #print(df)
        print(f"\033]0;Training (DONE)\a")

    if view:
        print("\033]0;Viewing\a")
        from torchview import draw_graph
        model_path = f"{M(1).__class__.__name__}_{mode}_{dataset_name}.model.pth"
        model_data = json.load(open(model_path.split(".")[0] + ".model.data.json"))
        net = M(n_features=model_data["n_features"], fig_size=28, n_channels=1, mode=net_mode)

        model_graph = draw_graph(net, input_size=(1, 28, 28),
                                 expand_nested=True,
                                 graph_name=model_path.split(".")[0],
                                 save_graph=True,
                                 directory="figs"
                                 )
        model_graph.visual_graph
        #input("Press Enter to continue...")

    if decode:
        print("\033]0;Decoding\a")
        import torch
        with torch.no_grad():
            if temp:
                model_path = "./model.temp.pth"
            elif "-lines" in sys.argv:
                mode="lines"
            elif "-dots" in sys.argv:
                mode="projected"
            else:
                mode = "double_connected"
            #print(M(1).__dict__)
            model_path = f"{M(1).__class__.__name__}_{mode}_{dataset_name}.model.pth"
            print(model_path)
            model_data = json.load(open(model_path.split(".")[0] + ".model.data.json"))

            n_features = model_data["n_features"]
            net = M(n_features=n_features, fig_size=model_data["fig_size"], n_channels=model_data["n_channels"])

            print("N FEATURES:", n_features)
            #print(model_data["index_to_label"])
            for n in range(n_features):
                print(" - ", n, model_data["index_to_label"][str(n)])
            net.load_state_dict(torch.load(model_path, weights_only=True))

            to_pred = []
            pred_labels = []
            preds = []
            decodes = []
            to_dec = []

            if "-c" in sys.argv:
                codes = [p for p in sys.argv[sys.argv.index("-c") + 1].split(",")]
                for code in codes:
                    i_path = f"imgs/{mode}/{code}.png"
                    image = Image.open(i_path)
                    image = transform(image)
                    print(image[-1])
                    image = torch.Tensor(np.array([image[-1]]))
                    print(image)
                    bits = net.forward(image)
                    to_pred.append(bits)
                    to_dec.append(bits)


                print(to_pred)
                print(len(to_pred))
            for p, d in zip(to_pred, to_dec):
                i = torch.Tensor(p)
                in_img = transforms.functional.to_pil_image(i, mode="L")
                plt.imshow(in_img)
                plt.colorbar()
                plt.show()
                plt.close()
                print(i.shape)

                k = torch.max(p, 1).indices[0].numpy()
                print("CLASS:", k, "VAL:", i[0,k])
                preds.append(model_data["index_to_label"][str(k)])
                print("PRED:", preds[-1])
                print("Decoding:", d)
                dec = net.backward(d)[0]
                print(dec)
                print(dec.shape)

                dec = transforms.functional.to_pil_image(dec, mode="L")

                decodes.append(dec)
                #print(decodes[-1])
                plt.imshow(dec)
                plt.colorbar()
                plt.show()

                #plt.show()



        plt.savefig(f"./{net.__class__.__name__}_lines_{dataset_name}_n{len(to_pred[0])}_p{"_".join(str(i) for i in pred_labels)}.png")





force = "-f" in sys.argv
if "-l" in sys.argv or "-e" in sys.argv:
    get_PCA(force=force, labs="-l"in sys.argv, images="-e" in sys.argv)
else:
    mode = "connected"
    if "double" in sys.argv:
        mode = "double_connected"
    image_classifier(mode=mode, train="-t" in sys.argv, decode="decode" in sys.argv, temp="temp" in sys.argv, view="view" in sys.argv)
