
import argparse
from pathlib import Path

import networkx as nx

import torch
import torch.nn as nn

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GATConv, GlobalAttention
from torch_geometric.utils import to_networkx, degree

import numpy as np

import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy.spatial import Delaunay

import math
import copy
import timeit
import os
from itertools import combinations

# Cut of the input graph


def cut(graph):
    cut = torch.sum((graph.x[graph.edge_index[0],
                             :2] != graph.x[graph.edge_index[1],
                                            :2]).all(axis=-1)).detach().item() / 2
    return cut

# Volumes of the partitions


def volumes(graph):
    ia = torch.where(
        (graph.x[:, :2] == torch.tensor([1.0, 0.0])).all(axis=-1))[0]
    ib = torch.where(
        (graph.x[:, :2] != torch.tensor([1.0, 0.0])).all(axis=-1))[0]
    degs = degree(
        graph.edge_index[0],
        num_nodes=graph.x.size(0),
        dtype=torch.uint8)
    da = torch.sum(degs[ia]).detach().item()
    db = torch.sum(degs[ib]).detach().item()
    cut = torch.sum((graph.x[graph.edge_index[0],
                             :2] != graph.x[graph.edge_index[1],
                                            :2]).all(axis=-1)).detach().item() / 2
    return cut, da, db

# Normalized cut of the input graph


def normalized_cut(graph):
    cut, da, db = volumes(graph)
    if da == 0 or db == 0:
        return 2
    else:
        return cut * (1 / da + 1 / db)

# Change the feature of the selected vertex


def change_vertex(state, vertex):
    if (state.x[vertex] == torch.tensor([1., 0.])).all():
        state.x[vertex] = torch.tensor([0., 1.])
    else:
        state.x[vertex] = torch.tensor([1., 0.])
    return state

# Reward to train the DRL agent


def reward_nc(state, vertex):
    new_state = state.clone()
    new_state = change_vertex(new_state, vertex)
    return normalized_cut(state) - normalized_cut(new_state)

# Networkx geometric Delaunay mesh with n random points in the unit square


def graph_delaunay_from_points(points):
    mesh = Delaunay(points, qhull_options="QJ")
    mesh_simp = mesh.simplices
    edges = []
    for i in range(len(mesh_simp)):
        edges += combinations(mesh_simp[i], 2)
    e = list(set(edges))
    return nx.Graph(e)

# Build a pytorch geometric graph with features [1,0] form a networkx
# graph. Then it turns the feature of one of the vertices with minimum
# degree into [0,1]


def torch_from_graph(g):

    adj_sparse = nx.to_scipy_sparse_matrix(g, format='coo')
    row = adj_sparse.row
    col = adj_sparse.col

    one_hot = []
    for i in range(g.number_of_nodes()):
        one_hot.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(one_hot), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    degs = np.sum(adj_sparse.todense(), axis=0)
    first_vertices = np.where(degs == np.min(degs))[0]
    first_vertex = np.random.choice(first_vertices)
    change_vertex(graph_torch, first_vertex)

    return graph_torch

# Training dataset made of Delaunay graphs generated from random points in
# the unit square


def delaunay_dataset(n, n_min, n_max):
    dataset = []
    for i in range(n):
        num_nodes = np.random.choice(np.arange(n_min, n_max + 1, 2))
        points = np.random.random_sample((num_nodes, 2))
        g = graph_delaunay_from_points(points)
        g_t = torch_from_graph(g)
        dataset.append(g_t)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader

# DRL training loop


def training_loop(
        model,
        training_dataset,
        gamma,
        time_to_sample,
        coeff,
        optimizer,
        print_loss
):

    i = 0

    # Here start the main loop for training
    for graph in training_dataset:

        if i % print_loss == 0 and i > 0:
            print('graph:', i, '  reward:', rew_partial)
        rew_partial = 0

        first_vertex = torch.where(graph.x == torch.tensor([0., 1.]))[
            0][0].item()

        start = graph

        len_episode = math.ceil(start.num_nodes / 2 - 1)  # length of an episod

        # this is the array that keeps track of vertices that have been flipped
        # yet
        S = [first_vertex]

        time = 0
        rew_partial = 0

        rews, vals, logprobs = [], [], []

        # Here starts the episod related to the graph "start"
        while time < len_episode:

            # we evaluate the A2C agent on the graph
            policy, values = model(start)
            probs = policy.view(-1).clone().detach().numpy()

            action = np.random.choice(np.arange(start.num_nodes), p=probs)

            S.append(action.item())

            rew = reward_nc(start, action)

            # Collect all the rewards in this episod
            rew_partial += rew

            # Collect the log-probability of the chosen action
            logprobs.append(torch.log(policy.view(-1)[action]))
            # Collect the value of the chosen action
            vals.append(values)
            # Collect the reward
            rews.append(rew)

            new_state = start.clone()
            # we flip the vertex returned by the policy
            new_state = change_vertex(new_state, action)
            # Update the state
            start = new_state

            time += 1

        # After time_to_sample episods we update the loss
        if i % time_to_sample == 0:  # and i > 0:

            logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
            vals = torch.stack(vals).flip(dims=(0,)).view(-1)
            rews = torch.tensor(rews).flip(dims=(0,)).view(-1)

            # Compute the advantage
            R = []
            R_partial = torch.tensor([0.])
            for j in range(rews.shape[0]):
                R_partial = rews[j] + gamma * R_partial
                R.append(R_partial)

            R = torch.stack(R).view(-1)
            advantage = R - vals.detach()

            # Actor loss
            actor_loss = (-1 * logprobs * advantage)

            # Critic loss
            critic_loss = torch.pow(R - vals, 2)

            # Finally we update the loss
            optimizer.zero_grad()

            loss = torch.mean(actor_loss) + \
                torch.tensor(coeff) * torch.mean(critic_loss)

            rews, vals, logprobs = [], [], []

            loss.backward()

            optimizer.step()

            rew_partial = 0  # restart the partial reward

        i += 1

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', default='./temp_edge/', type=str)
    parser.add_argument(
        "--nmin",
        default=50,
        help="Minimum graph size",
        type=int)
    parser.add_argument(
        "--nmax",
        default=100,
        help="Maximum graph size",
        type=int)
    parser.add_argument(
        "--ntrain",
        default=1000,
        help="Number of training graphs",
        type=int)
    parser.add_argument(
        "--print_rew",
        default=1000,
        help="Steps at which print reward",
        type=int)
    parser.add_argument("--batch", default=8, help="Batch size", type=int)
    parser.add_argument(
        "--lr",
        default=0.001,
        help="Learning rate",
        type=float)
    parser.add_argument(
        "--gamma",
        default=0.9,
        help="Gamma, discount factor",
        type=float)
    parser.add_argument(
        "--coeff",
        default=0.1,
        help="Critic loss coefficient",
        type=float)
    parser.add_argument(
        "--units_conv",
        default=[
            30,
            30,
            30,
            30],
        help="Number of units in conv layers",
        nargs='+',
        type=int)
    parser.add_argument(
        "--units_dense",
        default=[
            30,
            30,
            20],
        help="Number of units in linear layers",
        nargs='+',
        type=int)

    torch.manual_seed(1)
    np.random.seed(2)

    args = parser.parse_args()
    outdir = args.out + '/'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    n_min = args.nmin
    n_max = args.nmax
    n_train = args.ntrain
    coeff = args.coeff
    print_loss = args.print_rew

    time_to_sample = args.batch
    lr = args.lr
    gamma = args.gamma
    hid_conv = args.units_conv
    hid_lin = args.units_dense

    # Deep neural network that models the DRL agent

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = GATConv(2, hid_conv[0])
            self.conv2 = GATConv(hid_conv[0], hid_conv[1])
            self.conv3 = GATConv(hid_conv[1], hid_conv[2])
            self.conv4 = GATConv(hid_conv[2], hid_conv[3])

            self.l1 = nn.Linear(hid_conv[3], hid_lin[0])
            self.l2 = nn.Linear(hid_lin[0], hid_lin[1])
            self.actor1 = nn.Linear(hid_lin[1], hid_lin[2])
            self.actor2 = nn.Linear(hid_lin[2], 1)

            self.GlobAtt = GlobalAttention(
                nn.Sequential(
                    nn.Linear(
                        hid_lin[1], hid_lin[1]), nn.Tanh(), nn.Linear(
                        hid_lin[1], 1)))
            self.critic1 = nn.Linear(hid_lin[1], hid_lin[2])
            self.critic2 = nn.Linear(hid_lin[2], 1)

        def forward(self, graph):
            x_start, edge_index, batch = graph.x, graph.edge_index, graph.batch

            x = self.conv1(graph.x, edge_index)
            x = torch.tanh(x)
            x = self.conv2(x, edge_index)
            x = torch.tanh(x)
            x = self.conv3(x, edge_index)
            x = torch.tanh(x)
            x = self.conv4(x, edge_index)
            x = torch.tanh(x)

            x = self.l1(x)
            x = torch.tanh(x)
            x = self.l2(x)
            x = torch.tanh(x)

            x_actor = self.actor1(x)
            x_actor = torch.tanh(x_actor)
            x_actor = self.actor2(x_actor)
            flipped = torch.where(
                (x_start == torch.tensor([0., 1.])).all(axis=-1))[0]
            x_actor.data[flipped] = torch.tensor(-np.Inf)
            x_actor = torch.softmax(x_actor, dim=0)

            x_critic = self.GlobAtt(x, batch)
            x_critic = self.critic1(x_critic)
            x_critic = torch.tanh(x_critic)
            x_critic = self.critic2(x_critic)

            return x_actor, x_critic

    dataset = delaunay_dataset(n_train, n_min, n_max)

    model = Model()
    print(model)
    print('Model parameters:',
          sum([w.nelement() for w in model.parameters()]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    print('Start training')
    t0 = timeit.default_timer()
    model = training_loop(
        model,
        dataset,
        gamma,
        time_to_sample,
        coeff,
        optimizer,
        print_loss)
    ttrain = timeit.default_timer() - t0
    print('Training took:', ttrain, 'seconds')

    # Saving the model
    torch.save(model.state_dict(), outdir + 'model_coarsest')
