import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# column names of the dataset
names = ['Alcohol',
         'Malic acid',
         'Ash',
         'Alcalinity of ash',
         'Magnesium',
         'Total phenols',
         'Flavanoids',
         'Nonflavanoid phenols',
         'Proanthocyanins',
         'Color intensity',
         'Hue',
         'OD280/OD315 of diluted wines',
         'Proline'
         ]

# function for data preprocessing
def preprocess(data):
    labels = np.array(data.index) - 1
    attributes = data.reset_index(drop=True)

    # normalize the attributes
    attributes = (attributes - attributes.mean()) / attributes.std()
    return labels, attributes

# apply PCA to the dataset
def PCA_fit(attributes, var):
    pca_95 = PCA(n_components=var)
    pca_95.fit(attributes)
    n_components = pca_95.n_components_
    print("Number of principal components: ", n_components)
    attr_pca_95 = pca_95.transform(attributes)

    expl_var = pca_95.explained_variance_ratio_ * 100

    print("Total variance explained by {} components: {:.2f}%".format(
        n_components, sum(pca_95.explained_variance_ratio_)*100))
    return pca_95, attr_pca_95, expl_var, n_components

# plot percentage variance explained vs no. of PCs (cumulative)
def PCA_plot_var_vs_PC(expl_var):
    expl_var_cumsum = np.cumsum(expl_var)
    plt.plot(expl_var_cumsum)
    plt.title("Variance explained vs PC (cumulative)")
    plt.xlabel('PC')
    plt.xlim(0, 13)
    plt.ylabel('Explained variance')
    plt.savefig('./output_plots/pca_var_vs_PC.png')
    plt.clf()

# scatter plot wrt first two PCs
def PCA_plot_scatter(attr_pca_95, labels):
    plt.scatter(attr_pca_95[:, 0], attr_pca_95[:, 1],
                c=labels, s=5, cmap='coolwarm')
    plt.title("Data wrt first two PCs")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('./output_plots/pca.png')
    plt.clf()

# perform kmeans clustering to compute cluster assignments
def kmeans(X, y, k, random=False, max_iters=100, conv_thres=1e-6):

    N, n = X.shape
    cluster_reps, J_clus = [], []

    # if random = True, randomly initialize k cluster representatives
    if random:
        for i in range(k):
            cluster_reps.append(np.random.uniform(size=(n,)))

    # if random=False, choose k cluster representatives from given dataset
    else:
        idx_samples = np.random.choice(np.arange(N), k)
        cluster_reps = [X[idx] for idx in idx_samples]

    for iters in range(max_iters):

        # if decrease in J_clus < convergence threshold, stop
        if len(J_clus) > 1:
            if abs(J_clus[iters-1] - J_clus[iters-2]) < conv_thres:
                break

        # initialize group assignments
        clusters = np.zeros(N, dtype=np.uint8)
        groups = [[] for _ in range(k)]            # initialize groups
        cluster_labels = [[] for _ in range(k)]    # initialize cluster labels

        # do cluster assignments
        for i in range(N):
            clusters[i] = np.argmin(
                [np.linalg.norm(X[i] - cluster_reps[j], 2) for j in range(k)])
            groups[clusters[i]].append(X[i])
            cluster_labels[clusters[i]].append(y[i])

        # update cluster representatives
        for j in range(k):
            if len(groups[j]) == 0:
                cluster_reps[j] = np.zeros(n)
            else:
                cluster_reps[j] = np.mean(groups[j], axis=0)

        # calculate J_clus
        J = np.mean(np.square(
            [np.linalg.norm(X[i] - cluster_reps[clusters[i]], 2) for i in range(N)]))
        J_clus.append(J)

    # update representative label for each cluster
    repr_labels = -1*np.ones(k, dtype=np.uint8)
    for i in range(k):
        if len(cluster_labels[i]) > 0:
            repr_labels[i] = np.bincount(cluster_labels[i]).argmax()

    return clusters, cluster_reps, repr_labels, J_clus

# function to calculate NMI
def normalized_mutual_info(labels, clusters):

    # calculate mutual information
    mutual_info = mutual_information(labels, clusters)
    # calculate entropy
    entropy_labels = entropy(labels)
    entropy_clusters = entropy(clusters)
    # calculate normalized mutual information
    nmi = 2 * mutual_info / (entropy_labels + entropy_clusters)
    return nmi

# function to compute mutual information
def mutual_information(labels_true, labels_pred):

    mutual_info = 0
   
    pred_classes = np.unique(labels_pred, return_counts=True)    # classes in the predicted labels with their frequency
    true_classes = len(np.unique(labels_true))                   # number of classes in the true labels

    # calculate mutual information
    for i in range(len(pred_classes[0])):
        # probability of the class in the predicted labels
        p_class = np.zeros(true_classes)
        for j in range(len(labels_pred)):

            if labels_pred[j] == pred_classes[0][i]:
                p_class[labels_true[j]-1] += 1

        p_class /= pred_classes[1][i]
        # calculate the entropy in the class i
        Entropy = 0
        for j in range(true_classes):
            if p_class[j] != 0:
                Entropy += p_class[j] * np.log2(p_class[j])
        mutual_info += pred_classes[1][i] / len(labels_true) * Entropy
    # return H(Y) - H(Y|C)
    return entropy(labels_true) + mutual_info

# function to calculate entropy
def entropy(labels):

    entropy = 0
    N = len(labels)

    # calculate probability
    prob = np.unique(labels, return_counts=True)[1] / N
    # calculate entropy
    for p in prob:
        if p != 0:
            entropy += p * np.log2(p)
    return -entropy

# compute NMI for different values of k
def NMI_vs_k(k_start, k_end, X, y):
    NMI_vs_k = {}
    for k in range(k_start, k_end+1):
        clusters, _, _, _ = kmeans(X, y, k)
        NMI_vs_k[k] = normalized_mutual_info(clusters, y)
        print("NMI for k = {} is {}".format(k, NMI_vs_k[k]))
    return NMI_vs_k

# plot NMI vs k
def NMI_vs_k_plot(NMI_vs_k):
    plt.plot(list(NMI_vs_k.keys()), list(NMI_vs_k.values()))
    plt.title("NMI vs k")
    plt.xlabel('k')
    plt.ylabel('NMI')
    plt.savefig('./output_plots/NMI_vs_k.png')
    plt.clf()


if __name__ == "__main__":
    filename = 'wine.data'
    data = pd.read_csv(filename, sep=',', names=names)
    labels, attributes = preprocess(data)
    var = 0.95     # fractional value represents the percentage of variance to be retained

    print("Performing dimensionality reduction using PCA ...")
    pca_95, attr_pca_95, expl_var, n_components = PCA_fit(attributes, var)

    print("Plotting data wrt first two PCs...")
    PCA_plot_var_vs_PC(expl_var)
    PCA_plot_scatter(attr_pca_95, labels)
    print("PCA plots saved in ./output_plots/")

    k_start = 2
    k_end = 8
    print("Performing k-means clustering...")
    print("Computing NMI for k = {} to {} ...".format(k_start, k_end))
    NMI_vs_k = NMI_vs_k(k_start, k_end, attr_pca_95, labels)
    print("NMI computed.")
    print("Value of k for which NMI is maximum: {}".format(
        max(NMI_vs_k, key=NMI_vs_k.get)))
    print("max NMI: {}".format(NMI_vs_k[max(NMI_vs_k, key=NMI_vs_k.get)]))

    print("Plotting NMI vs k...")
    NMI_vs_k_plot(NMI_vs_k)
    print("NMI vs k plot saved in ./output_plots/")
