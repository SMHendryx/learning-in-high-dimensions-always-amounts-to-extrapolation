"""
Deep Interpolation Experiments

In this script, we reproduce recent findings from https://arxiv.org/pdf/2110.09485.pdf

And in particular, we look at interpolation as a function of the embedding dimension.

Balestriero et al. define interpolation as follows:
Definition 1. Interpolation occurs for a sample x whenever this sample belongs to the convex hull of a set of samples $\boldsymbol{X} \triangleq\left\{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right\}$, if not, extrapolation occurs.

Note that they don't define convex hulls per class, so we'll just define one convex hull for the entire training set regardless of class.
"""

from functools import partial

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
seed = 123
torch.manual_seed(seed)


# Model Adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, 10)
    
    def forward_embedding(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self.forward_embedding(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch, batch_size, log_interval, dry_run: bool = False, verbose: bool = True):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if verbose and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break
    if verbose: print(f"Train set accuracy over epoch {epoch}: {100. * correct / len(train_loader.dataset)}")


def test(model, device, test_loader, verbose: bool = True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbose: print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_model(embedding_dim: int = 128):
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 1.0
    gamma = 0.7
    dry_run = False
    log_interval = float("inf")
    save_model = False

    train_kwargs = {'batch_size': batch_size, "shuffle": True}
    test_kwargs = {'batch_size': test_batch_size, "shuffle": False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    log_interval = len(train_loader) - 1

    model = Net(embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in tqdm(range(1, epochs + 1)):
        train(model, device, train_loader, optimizer, epoch, batch_size, log_interval, dry_run, verbose=False)
        verbose = True if epoch == epochs else False
        test(model, device, test_loader, verbose)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    return model, train_loader, test_loader


def tensor_to_np_array(x):
    x = x.detach().cpu().numpy()
    return x

def gen_embeddings(model, dataloader):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            z = model.forward_embedding(data)
            z = tensor_to_np_array(z)
            all_embeddings.append(z)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def _solve_lp(x, c, A):
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def in_hull_lp(points, hull):
    """
    Solve linear program to determine if points are in the convex hull.
    """
    in_hulls = []
    n_dim = len(points[0])
    n_hull = len(hull)
    c = np.zeros(n_hull)
    A = np.r_[hull.T,np.ones((1,n_hull))]
    
    with mp.Pool(32) as p:
        in_hulls = p.map(partial(_solve_lp, c=c, A=A), points)

    return in_hulls



def percent_embeddings_in_hull(embeddings, hull):
    points_in_hull = in_hull_lp(embeddings, hull)
    return sum(points_in_hull) / len(embeddings) * 100.0


def percent_test_embeddings_in_train_hull(embedding_dim: int) -> float:
    model, train_loader, test_loader = train_model(embedding_dim)
    train_embeddings = gen_embeddings(model, train_loader)
    test_embeddings = gen_embeddings(model, test_loader)
    return percent_embeddings_in_hull(test_embeddings, train_embeddings)


def main():
    dims = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100, 128]
    percs = []
    for i, dim in enumerate(dims):
        print(f"Processing embedding dim: {dim}")
        perc = percent_test_embeddings_in_train_hull(dim)
        print(f"Percent of test embeddings in train hull of dimension {dim}: {perc}")
        percs.append(perc)
        df = pd.DataFrame({"embedding_dim": dims[:i + 1], "percent": percs})
        df.to_csv("results.csv")


if __name__ == "__main__":
    main()
