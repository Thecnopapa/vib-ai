import os, sys, json
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
        from models import SmallNet
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

        label_to_index = {}
        index_to_label = {}
        labs = []

        print("DATASET:", file_folder)
        print(f"Curating {len(os.listdir(file_folder))} images...")
        for n, file in enumerate(os.listdir(file_folder)):
            code = file.split(".")[0]
            l_path = os.path.join("labels", f"{code}.labels.json")
            #print(n, file, end=' label: ')
            if os.path.exists(l_path):
                lab_data = json.load(open(l_path))
                labs.extend([v["label"] for v in lab_data.values()] )
                structure_list.append(code)
                #print(l_path, end="\r")
            else:
                print(n, file, end=' label: ')
                print("Not found", end="\r")

        n_labs = {l: labs.count(l) for l in set(labs)}

        labs = list(set(labs))
        for n, l in enumerate(labs):
            label_to_index[str(l)] = int(n)
            index_to_label[int(n)] = str(l)


        class ImageDataset(Dataset):
            def __init__(self, struc_list, folder, label_folder=None):
                print("Loading data...")
                self.structures = struc_list
                self.folder = folder
                if label_folder is None:
                    self.label_folder = folder
                else:
                    self.label_folder = label_folder

                self.images = []
                self.labels = []
                self.image_dims = None
                for file in os.listdir(self.folder):
                    name = file.split(".")[0]
                    code, chain = name.split("_")
                    if code not in struc_list:
                        continue
                    #print(file, end="\r")
                    l_path = os.path.join(self.label_folder, f"{code}.labels.json")
                    if os.path.exists(l_path):
                        try:
                            lab = json.load(open(l_path))[chain]["label"]
                            self.labels.append(lab)
                            self.images.append(file)
                        except KeyError:
                            pass
                img = Image.open(os.path.join(self.folder, self.images[0]))
                #print(img)
                self.channels = 1
                self.image_dims = img.size[0]
                print(f"Loaded {len(self)} images!")




            def __len__(self):
                return len(self.images)



            def __getitem__(self, idx, as_image=False):
                fname = self.images[idx]
                name = os.path.basename(fname).split(".")[0]
                code, ch = name.split("_")

                i_path = os.path.join(self.folder, fname)
                l_path = os.path.join(self.label_folder, f"{code}.labels.json")

                label = label_to_index[json.load(open(l_path))[ch]["label"].lower().strip()]


                image = Image.open(i_path)
                #image = image.convert("RGB")
                if as_image:
                    return image, label
                image = transform(image)#[-1]#.resize((1,self.image_dims,self.image_dims))
                image = image[-1]

                # print("emb:", emb.shape, "lab:", lab)

                return image, label





        batch_size = 1
        print()
        print("N chains:", len(structure_list))
        print("N labs:", len(labs))
        print(labs)
        np.random.seed(6)
        train_list, test_list = train_test_split(structure_list, test_size=0.2, random_state=42)


        img_folder = f"imgs/{mode}"
        assert os.path.exists(img_folder)
        print("IMG_FOLDER:", img_folder)

        #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainset = ImageDataset(train_list, folder=img_folder, label_folder="labels")
        testset = ImageDataset(test_list, folder=img_folder, label_folder="labels")

        from parallel import cpu_count
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

        classes = labs

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

        def matplotlib_imshow(img, one_channel=False):
            if one_channel:
                img = img.mean(dim=0)
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            if one_channel:
                plt.imshow(npimg, cmap="Greys")
            else:
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
        import torchvision
        images = np.array([[trainset.__getitem__(x, as_image=False)[0].numpy()] for x in range(4)])
        #print(images)
        labels = [trainset[x][1] for x in range(4)]


        # create grid of images
        #img_grid = torchvision.utils.make_grid(images)

        # show images
        #matplotlib_imshow(img_grid, one_channel=False)

        # write to tensorboard
        #print(images, labels)
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


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        epochs = 13
        print(f"PRINT EVERY: {len(trainloader) // 10}")
        for epoch in range(epochs):  # loop over the dataset multiple times
            print(f"\033]0;Training (E={epoch+1}/{epochs})\a")

            torch.save(net.state_dict(), "./model.temp.pth")
            with open("./model.temp.data.json", "w") as f:
                json.dump(data, f, indent=4)
            if epoch//5 == 0:
                #image_classifier(train=False, decode=True, temp=True)
                pass

            running_loss = 0.0
            print(f'[{epoch + 1}, {0:5d}] loss: {running_loss / 10:.3f}', end="\r")
            for i, d in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = d

                # zero the parameter gradients
                optimizer.zero_grad()



                # forward + backward + optimize
                outputs = net(inputs)
                #print(outputs.shape)
                #print(inputs.shape)
                #print(labels.shape)
                #print(outputs)
                #print(labels)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                #if i < 4:
                #    #print(net.last_decode)
                #    #writer.add_image(f"output/train ({i})", net.last_decode[0], epoch)

                if i % (len(trainloader) // 10) == 0 and i != 0:  # print every 1000 mini-batches
                    print(f'[{epoch + 1}, {i + 0:5d}] loss: {running_loss / 10:.3f}', end = "\r")
                    writer.add_scalar("Loss/train", running_loss, epoch+1)
                    running_loss = 0.0

            with torch.no_grad():
                for i_name in ["1M2Z_A", "1AQK_L", "1BWW_A"]:
                    try:
                        i_path = f"imgs/{mode}/{i_name}.png"
                        image = Image.open(i_path)
                        image = transform(image)
                        image = torch.Tensor(np.array([image[-1]]))
                        bits = net.forward(image)
                        dec = net.decode(bits)[0]
                        #dec = transforms.functional.to_pil_image(dec, mode=None)
                        writer.add_image(f"test/in ({i_name})", image, epoch+1)
                        writer.add_image(f"test/out ({i_name})", dec, epoch+1)
                    except Exception as e:
                        print(e)
                        continue
        print()
        print('Finished Training ({} epochs)'.format(epochs))
        print(f"\033]0;Training (postprocess)\a")

        PATH = f'./{net.__class__.__name__}_{mode}_{dataset_name}.model.pth'

        torch.save(net.state_dict(), PATH)

        #dataiter = iter(testloader)
        #images, labels = next(dataiter)


        net = M(n_features = len(labs), n_chanels=trainset.channels, fig_size=trainset.image_dims)
        net.load_state_dict(torch.load(PATH, weights_only=True))

        #outputs = net(images)

        #_, predicted = torch.max(outputs, 1)

        correct = 0
        total = 0
        pres = []
        truths = []

        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i, l, p in zip(images, labels, predicted):
                    print(i, l, p)
                    pres.append(classes[l])
                    truths.append(classes[p])
                    if l == p:
                        correct_pred[classes[l]] += 1
                    total_pred[classes[l]] += 1

        print(f'Accuracy of {len(testset)}/{len(trainset)} test images: {100 * correct // total} %')


        from internship.plotting import plot_confusion
        print(zip(pres, truths))
        plot_confusion(truths, pres, f"{net.__class__.__name__}_{mode}_{dataset_name}", 100 * correct // total,  classes )



        if "mega" in sys.argv:
            dataset = "mega"
        else:
            dataset = "rcps"


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


        for classname, correct_count in sorted([(k,v) for k, v in correct_pred.items()], key=lambda x: n_labs[x[0]], reverse=True):
            #print(total_pred)
            df.sort_index(level="cath", inplace=True)
            title = get_family_desc(classname)
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

            df.loc[(classname, dataset, mode), "cath"] = classname
            df.loc[(classname, dataset, mode), "title"] = title
            df.loc[(classname, dataset, mode), "dataset"] = dataset
            df.loc[(classname, dataset, mode), "mode"] = mode

            df.loc[(classname, dataset, mode), "n_samples"] = n_labs[classname]

            if total_pred[classname] == 0:
                accuracy = None
            else:
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class ({label_to_index[classname]}): {classname:5s}: \t{accuracy:.1f}%\tcorrect: {correct_count}/{total_pred[classname]}\tin data: {n_labs[classname]}\ttitle: {title}')

                df.loc[(classname, dataset, mode), "correct"] = correct_count
                df.loc[(classname, dataset, mode), "total"] = total_pred[classname]

            df.loc[(classname, dataset, mode), "accuracy"] = accuracy
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
                mode = "connected"
            model_path = f"SmallNet_{mode}_{dataset_name}.model.pth"
            print(model_path)
            model_data = json.load(open(model_path.split(".")[0] + ".model.data.json"))

            n_features = model_data["n_features"]
            net = M(n_features=n_features, fig_size=model_data["fig_size"], n_chanels=model_data["n_channels"])

            print("N FEATURES:", n_features)
            print(model_data["index_to_label"])
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
                print(i.shape)


                preds.append(model_data["index_to_label"][str(torch.max(p, 1).indices[0].numpy())])
                print("PRED:", preds[-1])
                dec = net.decode(d)[0]
                print(dec)
                print(dec.shape)
                dec = transforms.functional.to_pil_image(dec, mode="L")

                decodes.append(dec)
                print(decodes[-1])
                plt.imshow(dec)
                plt.show()



        plt.savefig(f"./{net.__class__.__name__}_lines_{dataset_name}_n{len(to_pred[0])}_p{"_".join(str(i) for i in pred_labels)}.png")
        plt.show()




force = "-f" in sys.argv
if "-l" in sys.argv or "-e" in sys.argv:
    get_PCA(force=force, labs="-l"in sys.argv, images="-e" in sys.argv)
if "-t" in sys.argv:
    if "-all" in sys.argv:
        image_classifier(mode="connected")
        image_classifier(mode="projected")
        image_classifier(mode="lines")
    elif "-lines" in sys.argv:
        image_classifier(mode="lines")
    elif "-dots" in sys.argv:
        image_classifier(mode="projected")
    else:
        image_classifier(mode="double_connected")
if "decode" in sys.argv:
    image_classifier(mode="connected", train=False, decode=True, temp="temp" in sys.argv)
if "view" in sys.argv:
    image_classifier(mode="connected", train=False, view=True)










