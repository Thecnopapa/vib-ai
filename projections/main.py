import os, sys, json
import warnings
import numpy as np
import pandas as pd


from bioiain.utilities import str_to_list_with_literals

import PIL.Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from mpl_toolkits.axes_grid1 import ImageGrid

from parallel import *


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
    from sklearn.decomposition import PCA
    sys.path.append(".")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    if "mega" in sys.argv:
        file_folder = bi.biopython.downloadPDB("../internship/data", "mega-batch", file_path="../internship/data/mega-batch20K.txt", file_format="cif", overwrite=False)
    else:
        file_folder = bi.biopython.downloadPDB("../internship/data", "receptors", file_path="../internship/data/receptors.txt", file_format="cif", overwrite=False)
    batches = split_iterable(sorted(os.listdir(file_folder)))
    #[print(b) for b in batches]
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
            for chain in chains:
                projected_path = f"imgs/projected/{code}_{chain.id}.png"
                connected_path = f"imgs/connected/{code}_{chain.id}.png"
                lines_path = f"imgs/lines/{code}_{chain.id}.png"
                paths = (projected_path, connected_path, lines_path)
                if any([not os.path.exists(p) for p in paths]) or force:
                    coords = [a.coord for a in chain.get_atoms() if a.id == "CA"]
                    if len(coords) < 10:
                        continue
                    pca = PCA(n_components=3)
                    pca.fit(coords)
                    projected = pca.transform(coords)


                if (not os.path.exists(projected_path)) or (not os.path.exists(projected_path)) or force :
                    fig = plt.figure(figsize=(1.28,1.28))
                    ax = fig.add_subplot(111)

                    ax.scatter(projected[:, 0], projected[:, 1], c="#00000050", marker=".")
                    ax.set_aspect("equal")
                    ax.axis('off')

                    os.makedirs("imgs/projected", exist_ok=True)
                    os.makedirs("imgs/connected", exist_ok=True)
                    os.makedirs("imgs/lines", exist_ok=True)
                    os.makedirs("labels", exist_ok=True)



                    fig.savefig(projected_path, transparent=True)

                    for i in range(len(projected)-1):
                        ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000050")
                    fig.savefig(connected_path, transparent=True)

                    plt.close(fig)


                if (not os.path.exists(lines_path)) or force:
                    fig = plt.figure(figsize=(1.28, 1.28))
                    ax = fig.add_subplot(111)
                    ax.set_aspect("equal")
                    ax.axis('off')
                    for i in range(len(projected)-1):
                        ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000050")
                    fig.savefig(lines_path, transparent=True)
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
    from models import Net
    import torch
    import torch.nn as nn

    if "mega" in sys.argv:
        dataset_name = "mega"
        file_folder = bi.biopython.downloadPDB("../internship/data", "mega-batch",
                                               file_path="../internship/data/mega-batch20K.txt", file_format="cif",
                                               overwrite=False)
    else:
        dataset_name = "rcps"
        file_folder = bi.biopython.downloadPDB("../internship/data", "receptors",
                                               file_path="../internship/data/receptors.txt", file_format="cif",
                                               overwrite=False)

    if train:



        transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Resize((64,64)),
             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        structure_list = []

        label_to_index = {}
        index_to_label = {}
        labs = []

        print("DATASET:", file_folder)
        print(f"Curating {len(os.listdir(file_folder))} images...")
        for n, file in enumerate(os.listdir(file_folder)):
            code = file.split(".")[0]
            l_path = os.path.join("labels", f"{code}.labels.json")
            print(n, file, end='label: ')
            if os.path.exists(l_path):
                lab_data = json.load(open(l_path))
                labs.extend([v["label"] for v in lab_data.values()] )
                structure_list.append(code)
                print(l_path, end="\r")
            else:
                print("Not found", end="\r")

        n_labs = {l: labs.count(l) for l in set(labs)}

        labs = list(set(labs))
        for n, l in enumerate(labs):
            label_to_index[l] = n
            index_to_label[n] = l


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
                    print(file, end="\r")
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



            def __getitem__(self, idx):
                fname = self.images[idx]
                name = os.path.basename(fname).split(".")[0]
                code, ch = name.split("_")

                i_path = os.path.join(self.folder, fname)
                l_path = os.path.join(self.label_folder, f"{code}.labels.json")

                image = Image.open(i_path)
                #image = image.convert("RGB")
                image = transform(image)#[-1]#.resize((1,self.image_dims,self.image_dims))
                #print(image)
                #print(image.shape)
                if False:
                    print(i_path)

                    imgs = []
                    for channel in image:
                        imgs.append(T.ToPILImage()(channel))

                    fig = plt.figure(figsize=(8, 8))
                    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
                    for ax, im in zip(grid, imgs):
                        ax.imshow(im)
                        ax.set_axis_off()
                    plt.show()

                image = image[-1]

                label = label_to_index[json.load(open(l_path))[ch]["label"].lower().strip()]

                # print("emb:", emb.shape, "lab:", lab)
                return image, label





        batch_size = 1

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





        net = Net(len(labs), n_chanels=trainset.channels, fig_size=trainset.image_dims)

        import torch.optim as optim

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        epochs = 20
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("EPOCH: ", epoch, end="\r")
            torch.save(net.state_dict(), "./model.temp.pth")
            if epoch//5 == 0:
                #image_classifier(train=False, decode=True, temp=True)
                pass

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

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
                if i % 10 == 9:  # print every 1000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}', end = "\r")
                    running_loss = 0.0

        print('Finished Training ({} epochs)'.format(epochs))

        PATH = f'./{net.__class__.__name__}_{mode}_{dataset_name}.model.pth'

        torch.save(net.state_dict(), PATH)

        dataiter = iter(testloader)
        images, labels = next(dataiter)


        net = Net(n_features = len(labs), n_chanels=trainset.channels, fig_size=trainset.image_dims)
        net.load_state_dict(torch.load(PATH, weights_only=True))

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        correct = 0
        total = 0
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

        print(f'Accuracy of {len(testset)}/{len(trainset)} test images: {100 * correct // total} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    #print(label, prediction)
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


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

    if view:
        from torchview import draw_graph
        model_graph = draw_graph(Net(n_features=6, fig_size=128, n_chanels=1), input_size=(1, 1, 128, 128),
                                 expand_nested=True,
                                 graph_name="graph_1",
                                 save_graph=True,
                                 )
        model_graph.visual_graph
        input("Press Enter to continue...")

    if decode:
        import torch
        to_pred = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
        ]
        pred_labels = [0, 1, 2, 3, 4, 5]
        preds = []
        n_features = 6
        if "-n" in sys.argv:
            n_features = int(sys.argv[sys.argv.index("-n") + 1])
        if "-p" in sys.argv:

            pred_labels = [int(p) for p in sys.argv[sys.argv.index("-p") + 1].split(",")]
            to_pred = []
            for pred_label in pred_labels:
                to_pred.append([0]*pred_label+[1]+[0]*(n_features-pred_label-1))
            print(to_pred)
            print(len(to_pred))
        for p in to_pred:
            i = torch.Tensor(p)
            print(i.shape)
            net = Net(n_features=n_features, fig_size=128, n_chanels=1)
            if temp:
                model_path = "./model.temp.pth"
            elif "-lines" in sys.argv:
                model_path = f'./{net.__class__.__name__}_lines_{dataset_name}.model.pth'
            elif "-dots" in sys.argv:
                model_path = f'./{net.__class__.__name__}_projected_{dataset_name}.model.pth'
            else:
                model_path = f'./{net.__class__.__name__}_connected_{dataset_name}.model.pth'
            print(model_path)
            net.load_state_dict(torch.load(model_path, weights_only=True))
            decoded = net.decode(i)
            print("DECODED:")
            print(decoded)
            print(decoded.shape)

            print(decoded.shape)
            decoded = decoded.detach()
            perm = decoded
            #perm = perm.permute(0, 1, 2)
            perm = torch.sigmoid(perm) * 255
            print(perm)

            print(perm.shape)
            #plt.imshow(perm[0:2], alpha=perm[3] )
            #plt.imshow(np.squeeze(decoded.detach().numpy()))
            #plt.show()
            img = T.ToPILImage()(perm)
            preds.append(img)

        fig = plt.figure(figsize=(12, 8))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.1)
        for ax, im in zip(grid, preds):
            ax.imshow(im)
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
        image_classifier(mode="connected")
if "decode" in sys.argv:
    image_classifier(mode="connected", train=False, decode=True, temp="temp" in sys.argv)
if "view" in sys.argv:
    image_classifier(mode="connected", train=False, view=True)










