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
    if "mega" in sys.argv:
        file_folder = bi.biopython.downloadPDB("../internship/data", "mega-batch", file_path="../internship/data/mega-batch20K.txt", file_format="cif", overwrite=False)
    if "cath" in sys.argv:
        file_folder = bi.biopython.downloadPDB("../internship/data", "cath-nonredundant-S20",
                                                       file_path="../internship/data/cath-dataset-nonredundant-S20.list", file_format="cif",
                                                       overwrite=False)
    else:
        file_folder = bi.biopython.downloadPDB("../internship/data", "receptors", file_path="../internship/data/receptors.txt", file_format="cif", overwrite=False)
    batches = split_iterable(sorted(os.listdir(file_folder)))
    #[print(b) for b in batches]
    os.makedirs("labels", exist_ok=True)
    def generate_cath_labels(batch, do_labs, do_images):
        print("STARTING BATCH")
        for file in sorted(batch):
            code = file.split(".")[0]
            structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{code}.cif"))
            labels = {}
            chains = list(structure.get_chains())
            label_path = f"labels/{code}.labels.json"
            #print(structure, end=":\t")
            print(structure)
            if ((not os.path.exists(label_path)) or force) and do_labs:
                for chain in chains:
                    l = None
                    cath = parse_cath(code, chain.id)
                    if cath is not None:
                        l = f"{cath['class_number']}.{cath['arch_number']}.{cath['top_number']}.{cath['hom_fam_number']}"
                        labels[chain.id] = {
                            "chain_id": chain.id,
                            "cath": cath,
                            "label": l
                        }
                    #print(f"{chain.id}:{l}", end="\t")
                json.dump(labels, open(label_path, "w"), indent=4)
            #print("")

            if not do_images:
                continue

            os.makedirs("imgs", exist_ok=True)
            #if code not in ["1M2Z", "1P93"]:
            #    continue
            for chain in chains:
                projected_path = f"imgs/projected/{code}_{chain.id}.png"
                connected_path = f"imgs/connected/{code}_{chain.id}.png"
                double_path = f"imgs/double/{code}_{chain.id}.png"
                double_connected_path = f"imgs/double_connected/{code}_{chain.id}.png"
                paths = (projected_path, connected_path, double_path, double_connected_path)

                if any([not os.path.exists(p) for p in paths]) or force:

                    coords = [a.coord for a in chain.get_atoms() if a.id == "CA"]
                    if len(coords) < 5:
                        continue
                    for p in paths:
                        os.makedirs(os.path.dirname(p), exist_ok=True)

                    pca = PCA(n_components=3, random_state=6)
                    pca.fit(coords)

                    projected = pca.transform(coords)


                    # Single Projected

                    fig = plt.figure(figsize=(1.28,1.28))
                    ax = fig.add_subplot(111)
                    ax.set_aspect("equal")
                    ax.axis("off")

                    ax.scatter(projected[:, 0], projected[:, 1], c="#00000050", marker=".")

                    fig.savefig(projected_path, transparent=True)

                    # Single connected

                    for i in range(len(projected)-1):
                        ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000050")
                    fig.savefig(connected_path, transparent=True)

                    plt.close(fig)


                    # Double Projected

                    fig = plt.figure(figsize=(1.28, 1.28))
                    ax = fig.add_subplot(111)
                    ax.set_aspect("equal")
                    ax.axis('off')

                    ax.scatter(projected[:, 0], projected[:, 1], c="#00000025", marker=".")
                    ax.scatter(np.array(projected[:, 0])*-1, np.array(projected[:, 1])*-1, c="#00000025", marker=".")


                    fig.savefig(double_path, transparent=True)


                    # Double Connected
                    for i in range(len(projected)-1):
                        ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000025")
                        ax.plot(np.array(projected[i:i+2, 0])*-1, np.array(projected[i:i+2, 1])*-1, color="#00000025")
                    fig.savefig(double_connected_path, transparent=True)
                    plt.close(fig)

    pool = ThreadPool()
    for b in batches:
        print(len(b), b[0], b[-1])
        pool.add(generate_cath_labels, b, do_labs=labs, do_images=images)
    pool.start(wait=True)




def image_classifier(mode="double_connected", train = True, decode=False, view=False, temp=False):
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from models import Net as M
    import torch
    import torch.nn as nn

    if "small" in sys.argv:
        from models import SmallNet as SmallNet
        M = SmallNet

    elif "small2" in sys.argv:
        from models import SmallNetv2 as SmallNet
        M = SmallNet

    if "mega" in sys.argv:
        dataset_name = "mega"
        file_folder = bi.biopython.downloadPDB("../internship/data", "mega-batch",
                                               file_path="../internship/data/mega-batch20K.txt", file_format="cif",
                                               overwrite=False)
    elif "cath" in sys.argv:
        dataset_name = "cath"
        file_folder = bi.biopython.downloadPDB("../internship/data", "cath-nonredundant-S20",
                                               file_path="../internship/data/cath-dataset-nonredundant-S20.list", file_format="cif",
                                               overwrite=False)
    else:
        dataset_name = "rcps"
        file_folder = bi.biopython.downloadPDB("../internship/data", "receptors",
                                               file_path="../internship/data/receptors.txt", file_format="cif",
                                               overwrite=False)

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
        print(f"Filtering {len(os.listdir(file_folder))} PDBs...")
        valid = 0
        no_label = 0
        no_image = 0
        for n, file in enumerate(os.listdir(file_folder)):
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
        n_labs["other"] = 0

        index_to_label = {0: "other"}
        label_to_index = {"other": 0}
        for k, v in n_labs.items():
            if v < 10:
                label_to_index[str(k)] = 0
                n_labs["other"] +=1
            else:
                n = len(index_to_label)
                label_to_index[str(k)] = int(n)
                index_to_label[int(n)] = str(k)

        labs = index_to_label.values()
        all_labs = label_to_index.keys()


        class ImageDataset(Dataset):
            def __init__(self,   pdb_codes=None, img_folder=None, label_folder=None):
                print(f"ImageDataset: Loading data ({len(pdb_codes)} ids)")
                assert img_folder is not None
                if len(pdb_codes[0]) == 4:
                    self.has_chains = False
                    self.has_separated_chains = False
                elif len(pdb_codes[0]) == 5:
                    self.has_chains = True
                    self.has_separated_chains = False
                elif len(pdb_codes[0]) == 6:
                    self.has_chains = True
                    self.has_separated_chains = True
                    self.separator=pdb_codes[0][4]
                else:
                    raise Exception("ImageDataset: provided list is not made of pdb codes and/or chains")


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

                self.validate_input(pdb_codes)


                example_img = Image.open(os.path.join(self.img_folder, self.images[0]))
                #print(img)
                self.channels = 1
                self.image_dims = example_img.size[0]

                print(f"ImageDataset: loaded {len(self)} images from {len(self.codes)} pdbs and {len(self.chains)} chains!")


            def validate_input(self, codes=None):

                no_label = 0

                valid_chains = []
                valid_codes = []
                for file in os.listdir(self.img_folder):
                    name = file.split(".")[0]
                    code, chain = name.split("_")
                    if codes is not None:
                        if self.has_chains:
                            if self.has_separated_chains:
                                if not (f"{code}{self.separator}{chain}" in codes):
                                    continue
                            else:
                                if not (f"{code}{chain}" in codes):
                                    continue
                        else:
                            if not (code in codes):
                                continue


                    l_path = os.path.join(self.label_folder, f"{code}.labels.json")
                    if os.path.exists(l_path):
                        try:
                            lab = json.load(open(l_path))[chain]["label"]
                            self.labels.append(lab)
                            self.images.append(file)
                        except KeyError:
                            no_label += 1
                            continue
                    else:
                        no_label += 1
                        continue
                    valid_chains.append(f"{code}_{chain}")
                    valid_codes.append(code)
                if no_label > 0:
                    print(f"ImageDataset: (Warning) Some labels were not found: {no_label}")
                self.codes = sorted(set(valid_codes))
                self.chains = sorted(set(valid_chains))
                return self.chains

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx, as_image=False):
                fname = self.images[idx]
                name = os.path.basename(fname).split(".")[0]
                code, ch = name.split("_")

                i_path = os.path.join(self.img_folder, fname)
                l_path = os.path.join(self.label_folder, f"{code}.labels.json")

                label = label_to_index[json.load(open(l_path))[ch]["label"].lower().strip()]


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
        print()
        print("N chains:", len(structure_list))
        print("N labs (n>10):", len(labs))
        print("N labs (all):", len(all_labs))
        #print(json.dumps({k:v for k,v in n_labs.items() if v>=10}, indent=4))
        np.random.seed(6)


        img_folder = f"imgs/{mode}"

        assert os.path.exists(img_folder)
        print("IMG_FOLDER:", img_folder)
        structure_list = list(ImageDataset(structure_list, img_folder=img_folder, label_folder="labels").chains)
        print(len(structure_list))
        #print(structure_list[0:20])
        train_list, test_list = train_test_split(structure_list, train_size=0.8, test_size=0.2, random_state=42)
        print(len(train_list), len(test_list))
        trainset = ImageDataset(train_list, img_folder=img_folder, label_folder="labels")
        testset = ImageDataset(test_list, img_folder=img_folder, label_folder="labels")

        print(f"ACTUAL VALID DATA [(img+label)/chain]: \033[0;36m{len(trainset)+len(testset)}\033[0m test/train: {len(testset)}/{len(trainset)} ({len(testset)/len(trainset):.2f})")


        from parallel import cpu_count
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)


        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

        images = np.array([[trainset.__getitem__(x, as_image=False)[0].numpy()] for x in range(4)])
        #print(images)
        labels = [trainset[x][1] for x in range(4)]

        for n, (lab, img) in enumerate(zip(labels, images)):
            #print(lab, img.shape)
            writer.add_image(f"input/train ({n})", torch.Tensor(img), 0)




        net = M(len(labs), n_chanels=trainset.channels, fig_size=trainset.image_dims)
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

        #[print(p.shape, type(p)) for p in net.f_net.parameters()]
        probs = []
        criterion = nn.CrossEntropyLoss()

        #optimizer = optim.SGD(net.f_net.parameters(), lr=0.001)
        optimizer = optim.Adam(net.f_net.parameters(), lr=0.001)


        #i_criterion = nn.CrossEntropyLoss()
        from models import DiceLoss
        i_criterion = DiceLoss
        #i_optimizer = optim.SGD(net.r_net.parameters(), lr=0.001)
        i_optimizer = optim.Adam(net.r_net.parameters(), lr=0.001)
        epochs = 42
        splitsize = len(trainloader) // 10
        print(f"SPLITSIZE: {splitsize}")
        try:
            for epoch in range(epochs):  # loop over the dataset multiple times
                print(f"\033]0;Training (E={epoch+1}/{epochs})\a")

                torch.save(net.state_dict(), "./model.temp.pth")
                with open("./model.temp.data.json", "w") as f:
                    json.dump(data, f, indent=4)


                running_loss = 0.0
                running_i_loss = 0.0
                if os.environ.get("SLURM_CPUS_PER_TASK", None) is None:
                    print(f'[{epoch + 1}, {0:5d}] loss: {running_loss / 10:.3f}', end="\r")

                for i, d in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    imgs, labels = d
                    if len(imgs.shape) == 3:
                        imgs = imgs.reshape([1, *imgs.shape])

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    outputs = net(imgs)
                    pred = torch.max(outputs, 1).indices[0].numpy()
                    truth = [0.]*len(labs)
                    truth[labels[0]] = 1.
                    truth = torch.Tensor(truth)
                    #print()
                    #print("TRUTH:", truth)
                    #print(outputs, m, outputs[0][m], truth[m])

                    #print(outputs.shape, truth.shape)
                    #print(outputs, truth.numpy())
                    loss = criterion(outputs[0], truth)
                    loss.backward(retain_graph=True)

                    #print("LOSS:", loss.item())
                    #print(loss.item())

                    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                    i_optimizer.zero_grad()
                    i_outputs = net.backward(torch.Tensor([truth.numpy()]))[0]

                    #print(i_outputs[0].shape, imgs[0].shape, imgs[0, :, :-1, :-1].shape)
                    img = imgs#[0, :, :-1, :-1]
                    #print(img)
                    #img = nn.Softmax(dim=1)(img)
                    #out = nn.Softmax(dim=1)(i_outputs[0])
                    #print(img.shape)
                    #print(img)
                    #ax[0].imshow(img.detach().numpy()[0])

                    #(out.shape)
                    #print(out)
                    #ax[1].imshow(out.detach().numpy()[0])
                    #plt.show()

                    #print(img)
                    #print(i_outputs[0])
                    i_loss = i_criterion(i_outputs[0], img[0])
                    #print("I-LOSS:", i_loss.item())
                    #print(i_loss.item())
                    i_loss.backward()
                    optimizer.step()
                    i_optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    running_i_loss += i_loss.item()
                    #if i < 4:
                    #    #print(net.last_decode)
                    #    #writer.add_image(f"output/train ({i})", net.last_decode[0], epoch)
                    print(
                        f'[{epoch + 1:2d}, {i%(splitsize+1):5d}] loss: {running_loss / (i % splitsize + 1):5.3f} i-loss: {running_i_loss / (i % splitsize+1):5.3f}',
                        end="\r")

                    if i % splitsize == 0 and i != 0:  # print every 1000 mini-batches
                        print(f'[{epoch + 1:2d}, {i:5d}] loss: {running_loss / splitsize:5.3f} i-loss: {running_i_loss / splitsize:5.3f}', end = "\n")
                        writer.add_scalar("loss/encode", running_loss / splitsize, epoch+1)
                        writer.add_scalar("loss/decode", running_i_loss / splitsize, epoch + 1)
                        running_loss = 0.0
                        running_i_loss = 0.0

                with torch.no_grad():
                    for i_name in ["1M2Z_A", "1AQK_L", "1BWW_A", "1GLU_A"]:
                        try:
                            i_path = f"imgs/{mode}/{i_name}.png"
                            image = Image.open(i_path)
                            image = transform(image)
                            image = torch.Tensor(np.array([image[-1]]))
                            #print(image)

                            #image = torch.sigmoid(image)
                            bits = net(image)
                            #print("BITS:", bits[0])
                            dec = net.backward(bits[0])
                            #image = nn.Softmax(dim=1)(image)#*256
                            #dec = nn.Softmax(dim=1)(dec)#*256
                            #print(dec.shape)
                            #print(dec)
                            #print(image.shape)
                            #print(image)
                            #dec = transforms.functional.to_pil_image(dec, mode=None)
                            #overlay = torch.cat((dec, image))
                            #print(overlay)
                            #print(overlay.shape)
                            #writer.add_image(f"test/{i_name} (overlay)", image, epoch + 1)

                            writer.add_image(f"test/{i_name} (in)", image, epoch+1)
                            writer.add_image(f"test/{i_name} (out)", dec, epoch+1)
                        except Exception as e:
                            print(e)
                            #exit()
                            continue
                    for l in labs:
                        try:
                            bits = [0]*len(labs)
                            #print(bits)
                            n = label_to_index[l]
                            bits[n] = 1
                            #print(bits)
                            bits = torch.Tensor(bits)
                            #print("BITS LAB:", bits)
                            dec = net.backward(bits)
                            #print(dec)
                            #print(dec.shape)
                            writer.add_image(f"classes/{l}", dec, epoch+1)
                        except Exception as e:
                            print(e)
                            exit()
                            continue
            print()
            print('Finished Training ({} epochs)'.format(epochs))
            print(f"\033]0;Training (postprocess)\a")

            PATH = f'./{net.__class__.__name__}_{mode}_{dataset_name}.model.pth'

            torch.save(net.state_dict(), PATH)
        except KeyboardInterrupt:
            PATH = f'./model.temp.pth'



        #dataiter = iter(testloader)
        #images, labels = next(dataiter)

        try:
            net = M(n_features = len(labs), n_chanels=trainset.channels, fig_size=trainset.image_dims)
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
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                #total += labels.size(0)
                #correct += (predicted == labels).sum().item()
                for i, l, p in zip(images, labels, predicted):
                    #print(i, l, p)
                    pres.append(index_to_label[int(l)])
                    truths.append(index_to_label[int(p)])

                    if l == p:
                        correct_pred[index_to_label[int(l)]] += 1
                        correct += 1
                    total_pred[index_to_label[int(l)]] += 1
                    total += 1

        print(f'Accuracy of {len(testset)}/{len(trainset)} test images: {100 * correct // total} %')


        from internship.plotting import plot_confusion
        plot_confusion(truths, pres, f"{net.__class__.__name__}_{mode}_{dataset_name}", 100 * correct // total, sorted(labs))



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
            columns = ["title", "cath",  "dataset", "mode"] + columns
            df = pd.DataFrame(columns=columns)


        df.set_index(["cath", "dataset", "mode"], inplace=True, drop=False)
        df.sort_index(level="cath", inplace=True)

        #print(df)
        #print("LEXSORTED:", df.index.is_monotonic_increasing, )


        for classname, correct_count in sorted([(k,v) for k, v in correct_pred.items() if k in labs], key=lambda x: n_labs[x[0]], reverse=True):
            #print(total_pred)
            df.sort_index(level="cath", inplace=True)
            title = get_family_desc(classname)
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

            df.loc[(classname, dataset_name, mode), "cath"] = classname
            df.loc[(classname, dataset_name, mode), "title"] = title
            df.loc[(classname, dataset_name, mode), "dataset"] = dataset_name
            df.loc[(classname, dataset_name, mode), "mode"] = mode

            df.loc[(classname, dataset_name, mode), "n_samples"] = n_labs[classname]

            if total_pred[classname] == 0:
                accuracy = None
                print(f'Accuracy for class ({label_to_index[classname]:3d}) -> {classname:12s}: \t{accuracy}\t(not in test set)\tin data: {n_labs[classname]:3d}\ttitle: {title}')

            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class ({label_to_index[classname]:3d}) -> {classname:12s}: \t{accuracy:4.2f}%\tcorrect:  {correct_count:3d}/{total_pred[classname]:3d}\tin data: {n_labs[classname]:3d}\ttitle: {title}')

                df.loc[(classname, dataset_name, mode), "correct"] = correct_count
                df.loc[(classname, dataset_name, mode), "total"] = total_pred[classname]

            df.loc[(classname, dataset_name, mode), "accuracy"] = accuracy
            warnings.simplefilter(action='default', category=pd.errors.PerformanceWarning)
            df.sort_index(level="cath", inplace=True)

        df.sort_index(level="cath", inplace=True)
        df.to_csv(df_path, index=False)
        print(df)
        print(f"\033]0;Training (DONE)\a")

    if view:
        print("\033]0;Viewing\a")
        from torchview import draw_graph
        model_graph = draw_graph(M(n_features=6, fig_size=128, n_chanels=1), input_size=(1, 1, 128, 128),
                                 expand_nested=True,
                                 graph_name="graph_1",
                                 save_graph=True,
                                 )
        model_graph.visual_graph
        input("Press Enter to continue...")

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
            print(M(1).__dict__)
            model_path = f"{M(1).__class__.__name__}_{mode}_{dataset_name}.model.pth"
            print(model_path)
            model_data = json.load(open(model_path.split(".")[0] + ".model.data.json"))

            n_features = model_data["n_features"]
            net = M(n_features=n_features, fig_size=model_data["fig_size"], n_chanels=model_data["n_channels"])

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
    image_classifier(mode="double_connected", train="-t" in sys.argv, decode="decode" in sys.argv, temp="temp" in sys.argv, view="view" in sys.argv)











