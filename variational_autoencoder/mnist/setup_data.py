import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# for loading and batching MNIST dataset
# - we use transforms.ToTensor() to normalize the pixel intensities to the range [0.0,1.0].
def setup_data_loaders(batch_size = 128, subset = False):
    root = '/Users/ricard/test/pyro/files'

    train_set = dset.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=True)
    test_set = dset.MNIST(root=root, train=False, transform=transforms.ToTensor())

    if batch_size=="full":
        X_train, y_train = get_all_data(train_set, num_workers=2, shuffle=False)[0]
        X_test, y_test = get_all_data(test_set, num_workers=2, shuffle=False)[0]
        return X_train, y_train, X_test, y_test

    if subset:
        batch_size = 100
        train_set = torch.utils.data.Subset(train_set, list(range(0,500)))
        test_set = torch.utils.data.Subset(test_set, list(range(0,500)))

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader

def get_all_data(dataset, num_workers=2, shuffle=False):
    dataset_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=dataset_size, num_workers=num_workers, shuffle=shuffle)
    data_concatenated = [ (batch[0], batch[1]) for (_, batch) in enumerate(data_loader) ]
    return data_concatenated

def plot_tsne(Z, classes, outdir):
    """ Not working """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    matplotlib.use('Agg')

    print("Running t-SNE...")
    model_tsne = TSNE(n_components=2, perplexity=30.0, init="random", n_iter=250, random_state=0)
    z_embed = model_tsne.fit_transform(Z)

    classes = classes[:,None]

    # Plot
    fig = plt.figure()
    for i in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, i] = 1
        ind_class = classes[:, i] == 1
        color = plt.cm.Set1(i)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        # fig.savefig(outdir+'/tsne_embedding_'+str(i)+'.png')
    plt.title("Latent Variable T-SNE per Class")
    fig.savefig(outdir+'/tsne_embedding.png')


def plot_elbo(train_elbo, test_elbo, file):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    data = np.concatenate([np.arange(len(test_elbo))[:,np.newaxis], train_elbo[:,np.newaxis], test_elbo[:,np.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Train ELBO', 'Test ELBO'])
    df = pd.melt(df, id_vars='Training Epoch', value_vars=['Train ELBO', 'Test ELBO'], var_name="type", value_name='elbo')
    sns.relplot(x="Training Epoch", y="elbo", hue="type", kind="line", data=df)
    plt.savefig(file)
    plt.close('all')
    # plt.show()
