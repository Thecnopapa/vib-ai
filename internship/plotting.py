
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
import bioiain as bi

def plot_embeddings(dataset, labels, title, score):
    bi.log(1, "Plotting embeddings...")
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform([dataset[d][0] for d in  range(len(dataset))])
    plt.figure(figsize=(12, 10))
    sb.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels, palette="Set1", s=40, alpha=0.8)
    plt.title(title)
    plt.savefig(f"figs/{title}_embeddings.png")



# Confusion matrices
def plot_confusion(preds, labels, title, score, classes):
    bi.log(1, "Plotting confusion...")
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8 ,8))
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title}_S={score:.3f}")
    plt.savefig(f"figs/{title}_confusion.png")