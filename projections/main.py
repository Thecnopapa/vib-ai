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





def get_PCA():
    from sklearn.decomposition import PCA
    import plotly.graph_objs as go
    import matplotlib.pyplot as plt
    file_folder = bi.biopython.downloadPDB("../internship/data", "receptors", file_path="../internship/data/receptors.txt", file_format="cif", overwrite=False)

    for file in os.listdir(file_folder):
        code = file.split(".")[0]
        structure = bi.biopython.loadPDB(os.path.join(file_folder, f"{code}.cif"))
        print(structure)
        for chain in structure[0].get_chains():

            coords = [a.coord for a in chain.get_atoms() if a.id == "CA"]
            if len(coords) < 10:
                continue
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
            os.makedirs("imgs/projected", exist_ok=True)
            os.makedirs("imgs/connected", exist_ok=True)

            fig.savefig(f"imgs/projected/{code}_{chain.id}.png")

            for i in range(len(projected)-1):
                ax.plot(projected[i:i+2, 0], projected[i:i+2, 1], color="#00000050")
            fig.savefig(f"imgs/connected/{code}_{chain.id}.png")
            # Pad the saved area by 10% in the x-direction and 20% in the y-direction
            #plt.show(block=False)
            plt.clf()

            # import PIL
            # img = PIL.Image.open(f"{code}_connected.png")
            # img = img.convert("RGB")
            #
            # import torchvision as tv
            # from torchvision.utils import save_image
            # tensor = tv.transforms.functional.pil_to_tensor(img)
            # print(tensor)
            # print(tensor.shape)
            # #save_image(tensor[1], f"{code}_tensor2.png")
            # plt.imshow(tensor)
            # #arr = np.ndarray(tensor.numpy())  # This is your tensor
            # #arr_ = np.squeeze(arr)  # you can give axis attribute if you wanna squeeze in specific dimension
            # #plt.imshow(arr_)
            # plt.savefig(f"{code}_tensor.png")


def image_classifier():
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    import matplotlib.pyplot as plt
    import numpy as np

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
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
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')



get_PCA()
#image_classifier()










