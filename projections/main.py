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





def get_PCA(force=False):
    from sklearn.decomposition import PCA
    import plotly.graph_objs as go
    import matplotlib.pyplot as plt
    file_folder = bi.biopython.downloadPDB("../internship/data", "mega-batch", file_path="../internship/data/mega-batch20K.txt", file_format="cif", overwrite=False)

    for file in sorted(os.listdir(file_folder)):
        code = file.split(".")[0]
        structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{code}.cif"))
        header = bioiain.biopython.imports.read_mmcif(os.path.join(file_folder, file), subset=["_entity_poly", "_entity"])
        labels = {}
        chains = list(structure.get_chains())

        if type(header["_entity_poly"]) is dict:
            poly = [header["_entity_poly"]]
        else:
            poly = header["_entity_poly"]
        for i, entity_poly in enumerate(poly):
            bi.log(1, i)
            #print(entity_poly)
            #print(entity_poly["pdbx_strand_id"])


            for strand in entity_poly["pdbx_strand_id"].split(","):
                strand = strand.strip()
                bi.log(2, strand)
                #print(chains[0].__dict__.keys())
                #print(chains[0]._id, chains[0].id, chains[0].full_id )
                #print(strand, chains, strand in chains)
                if strand in [c.id for c in chains]:
                    labels[strand] = {"description": header["_entity",i,"pdbx_description"],
                                      "length": [len(list(c.get_residues())) for c in chains if c.id == strand][0],
                                      "chain":list([c for c in chains if c.id == strand])[0]}

        #print(labels)






        for ch, l in labels.items():
            chain = l["chain"]
            if os.path.exists(f"labels/{code}_{chain.id}.labels.json") and not force:
                continue
            #print(l)
            #print(type(l["chain"]))

            label = l["description"]
            #print(chain, label)
            coords = [a.coord for a in chain.get_atoms() if a.id == "CA"]
            if len(coords) < 10:
                continue
            #print(coords[:10])
            #print(len(coords))
            pca = PCA(n_components=3)
            pca.fit(coords)
            #print(pca.components_)

            projected = pca.transform(coords)
            #print(projected[0:10])

            #print(bi)
            #from bioiain import visualisation
            fig = plt.figure(figsize=(1,1))
            ax = fig.add_subplot(111)

            ax.scatter(projected[:, 0], projected[:, 1], c="black", marker=".")
            ax.set_aspect("equal")
            ax.axis('off')

            os.makedirs("imgs/projected", exist_ok=True)
            os.makedirs("imgs/connected", exist_ok=True)
            os.makedirs("labels", exist_ok=True)

            projected_path = f"imgs/projected/{code}_{chain.id}.png"
            connected_path = f"imgs/connected/{code}_{chain.id}.png"

            fig.savefig(projected_path)

            for i in range(len(projected)-1):
                ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000050")
            fig.savefig(connected_path)

            plt.clf()

            exp = {
                "label": label,
                "file": file,
                "chain": chain.id,
                "paths": {
                    "connected": connected_path,
                    "projected": projected_path
                }
            }
            json.dump(exp, open(f"labels/{code}_{chain.id}.labels.json", "w"), indent=4)


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets.vision as vision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split



def image_classifier():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32,32)),
         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    structure_list = []

    label_to_index = {}
    index_to_label = {}
    labs = []
    for file in os.listdir("imgs/connected"):
        name = file.split(".")[0]
        l_path = os.path.join("labels", f"{name}.labels.json")
        if os.path.exists(l_path):
            labs.append(json.load(open(l_path))["label"].lower().strip())
            structure_list.append(name)
    labs = list(set(labs))
    for n, l in enumerate(labs):
        label_to_index[l] = n
        index_to_label[n] = l

    class ImageDataset(Dataset):
        def __init__(self, struc_list, folder, label_folder=None):
            self.structures = struc_list
            self.folder = folder
            if label_folder is None:
                self.label_folder = folder
            else:
                self.label_folder = label_folder

            self.images = []
            self.labels = []
            for file in os.listdir(self.folder):
                name = file.split(".")[0]
                if name not in struc_list:
                    continue
                l_path = os.path.join(self.label_folder, f"{name}.labels.json")
                if os.path.exists(l_path):
                    labs.append(json.load(open(l_path))["label"])
                    self.labels.append(f"{name}.labels.json")
                    self.images.append(file)


        def __len__(self):
            return len(self.images)



        def __getitem__(self, idx):
            fname = self.images[idx]
            name = os.path.basename(fname).split(".")[0]
            code, ch = name.split("_")

            i_path = os.path.join(self.folder, fname)
            l_path = os.path.join(self.label_folder, f"{name}.labels.json")

            image = Image.open(i_path)
            #image = image.convert("RGB")
            image = transform(image)

            label = label_to_index[json.load(open(l_path))["label"].lower().strip()]

            # print("emb:", emb.shape, "lab:", lab)
            return image, label





    batch_size = 4

    print(len(structure_list))
    print(labs)

    train_list, test_list = train_test_split(structure_list, test_size=0.2, random_state=42)


    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = ImageDataset(train_list, folder="imgs/connected", label_folder="labels")
    testset = ImageDataset(test_list, folder="imgs/connected", label_folder="labels")


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(trainset)
    print(trainloader)

    classes = labs

    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    # print(len(trainset))
    # t = []
    # for i in range(4):
    #     t.append([trainset[i]])
    #
    # images, labels = zip([[z[0], z[1]] for z in t])
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self, n_features=10):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, n_features)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net(len(labs))

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):  # loop over the dataset multiple times
        print("EPOCH: ", epoch)

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{index_to_label[int(labels[j])]:5s}' for j in range(4)))

    net = Net(n_features = len(labs))
    net.load_state_dict(torch.load(PATH, weights_only=True))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' / '.join(f'{index_to_label[int(predicted[j])]:5s}'
                                  for j in range(4)))

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

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

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

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        #print(total_pred)
        if total_pred[classname] == 0:
            accuracy = 999
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if "-l" in sys.argv or "-e" in sys.argv:
    get_PCA()
if "-t" in sys.argv:
    image_classifier()










