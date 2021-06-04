
import argparse
from pathlib import Path

import networkx as nx
import nxmetis

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, graclus, avg_pool, global_mean_pool
from torch_geometric.utils import to_networkx, k_hop_subgraph, degree

import numpy as np
from numpy import random

import scipy
from scipy.sparse import coo_matrix, rand
from scipy.io import mmread
from scipy.spatial import Delaunay

import copy
import timeit
import os
from itertools import combinations


# Networkx geometric Delaunay mesh with n random points in the unit square
def graph_delaunay_from_points(points):
    mesh = Delaunay(points, qhull_options="QJ")
    mesh_simp = mesh.simplices
    edges = []
    for i in range(len(mesh_simp)):
        edges += combinations(mesh_simp[i], 2)
    e = list(set(edges))
    return nx.Graph(e)

# Pytorch geometric Delaunay mesh with n random points in the unit square


def random_delaunay_graph(n):
    points = np.random.random_sample((n, 2))
    g = graph_delaunay_from_points(points)
    return torch_from_graph(g)

# Build a pytorch geometric graph with features [1,0] form a networkx graph


def torch_from_graph(graph):
    adj_sparse = nx.to_scipy_sparse_matrix(graph, format='coo')
    row = adj_sparse.row
    col = adj_sparse.col

    features = []
    for i in range(graph.number_of_nodes()):
        features.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(features), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    return graph_torch

# Training dataset made of Delaunay graphs generated from random points in
# the unit square and their coarser graphs


def delaunay_dataset_with_coarser(n, n_min, n_max):
    dataset = []
    while len(dataset) < n:
        number_nodes = np.random.choice(np.arange(n_min, n_max + 1, 2))
        g = random_delaunay_graph(number_nodes)
        dataset.append(g)
        while g.num_nodes > 200:
            cluster = graclus(g.edge_index)
            coarse_graph = avg_pool(
                cluster,
                Batch(
                    batch=torch.zeros(
                        g.num_nodes),
                    x=g.x,
                    edge_index=g.edge_index))
            g1 = Data(x=coarse_graph.x, edge_index=coarse_graph.edge_index)
            dataset.append(g1)
            g = g1

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader

# Build a pytorch geometric graph with features [1,0] form a sparse matrix


def torch_from_sparse(adj_sparse):

    row = adj_sparse.row
    col = adj_sparse.col

    features = []
    for i in range(adj_sparse.shape[0]):
        features.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(features), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    return graph_torch

# Training dataset made of SuiteSparse graphs and their coarser graphs


def suitesparse_dataset_with_coarser(n, n_min, n_max):
    dataset, picked = [], []
    for graph in os.listdir(os.path.expanduser(
            'drl-graph-partitioning/suitesparse_train/')):

        if len(dataset) > n or len(picked) >= len(os.listdir(os.path.expanduser(
                'drl-graph-partitioning/suitesparse_train/'))):
            break
        picked.append(str(graph))
        # print(str(graph))
        matrix_sparse = mmread(
            os.path.expanduser(
                'drl-graph-partitioning/suitesparse_train/' +
                str(graph)))
        gnx = nx.from_scipy_sparse_matrix(matrix_sparse)
        if nx.number_connected_components(gnx) == 1 and gnx.number_of_nodes(
        ) > n_min and gnx.number_of_nodes() < n_max:
            g = torch_from_sparse(matrix_sparse)
            g.weight = torch.tensor([1] * g.num_edges)
            dataset.append(g)
            while g.num_nodes > 200:
                cluster = graclus(g.edge_index)
                coarse_graph = avg_pool(
                    cluster,
                    Batch(
                        batch=torch.zeros(
                            g.num_nodes),
                        x=g.x,
                        edge_index=g.edge_index))
                g1 = Data(x=coarse_graph.x, edge_index=coarse_graph.edge_index)
                dataset.append(g1)
                g = g1

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader

# Cut of the input graph


def cut(graph):
    cut = torch.sum((graph.x[graph.edge_index[0],
                             :2] != graph.x[graph.edge_index[1],
                                            :2]).all(axis=-1)).detach().item() / 2
    return cut

# Change the feature of the selected vertex


def change_vertex(state, vertex):
    if (state.x[vertex, :2] == torch.tensor([1., 0.])).all():
        state.x[vertex, 0] = torch.tensor(0.)
        state.x[vertex, 1] = torch.tensor(1.)
    else:
        state.x[vertex, 0] = torch.tensor(1.)
        state.x[vertex, 1] = torch.tensor(0.)

    return state

# Reward to train the DRL agent


def reward_NC(state, vertex):
    new_state = state.clone()
    new_state = change_vertex(new_state, vertex)
    return normalized_cut(state) - normalized_cut(new_state)

# Normalized cut of the input graph


def normalized_cut(graph):
    cut, da, db = volumes(graph)
    if da == 0 or db == 0:
        return 2
    else:
        return cut * (1 / da + 1 / db)

# Coarsen a pytorch geometric graph, then find the cut with METIS and
# interpolate it back


def partition_metis_refine(graph):
    cluster = graclus(graph.edge_index)
    coarse_graph = avg_pool(
        cluster,
        Batch(
            batch=graph.batch,
            x=graph.x,
            edge_index=graph.edge_index))
    coarse_graph_nx = to_networkx(coarse_graph, to_undirected=True)
    _, parts = nxmetis.partition(coarse_graph_nx, 2)
    mparts = np.array(parts)
    coarse_graph.x[np.array(parts[0])] = torch.tensor([1., 0.])
    coarse_graph.x[np.array(parts[1])] = torch.tensor([0., 1.])
    _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
    graph.x = coarse_graph.x[inverse]
    return graph

# Subgraph around the cut


def k_hop_graph_cut(graph, k):
    nei = torch.where((graph.x[graph.edge_index[0], :2] !=
                       graph.x[graph.edge_index[1], :2]).all(axis=-1))[0]
    neib = graph.edge_index[0][nei]
    data_cut = k_hop_subgraph(neib, k, graph.edge_index, relabel_nodes=True)
    data_small = k_hop_subgraph(
        neib,
        k - 1,
        graph.edge_index,
        relabel_nodes=True)
    nodes_boundary = list(
        set(data_cut[0].numpy()).difference(data_small[0].numpy()))
    boundary_features = torch.tensor([1. if i.item(
    ) in nodes_boundary else 0. for i in data_cut[0]]).reshape(data_cut[0].shape[0], 1)
    _, va, vb = volumes(graph)
    e = torch.ones(data_cut[0].shape[0], 1)
    nnz = graph.num_edges
    features = torch.cat((graph.x[data_cut[0]], boundary_features, torch.true_divide(
        va, nnz) * e, torch.true_divide(vb, nnz) * e), 1)

    g_red = Batch(
        batch=torch.zeros(
            data_cut[0].shape[0],
            dtype=torch.long),
        x=features,
        edge_index=data_cut[1])
    return g_red, data_cut[0]

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


# Training loop
def training_loop(
        t,
        model,
        training_dataset,
        episodes,
        gamma,
        time_to_sample,
        coeff,
        optimizer,
        print_loss,
        k):

    # Here start the main loop for training
    for i in range(episodes):
        rew_partial = 0
        p = 0
        for graph in training_dataset:
            start_all = partition_metis_refine(graph)

            data = k_hop_graph_cut(start_all, k)
            graph_cut, positions = data[0], data[1]
            len_episode = cut(graph)

            start = graph_cut
            time = 0

            rews, vals, logprobs = [], [], []

            # Here starts the episod related to the graph "start"
            while time < len_episode:

                # we evaluate the A2C agent on the graph
                policy, values = model(start)
                probs = policy.view(-1)

                action = torch.distributions.Categorical(
                    logits=probs).sample().detach().item()

                # compute the reward associated with this action
                rew = reward_NC(start_all, positions[action])
                rew_partial += rew
                # Collect the log-probability of the chosen action
                logprobs.append(policy.view(-1)[action])
                # Collect the value of the chosen action
                vals.append(values)
                # Collect the reward
                rews.append(rew)

                new_state = start.clone()
                new_state_orig = start_all.clone()
                # we flip the vertex returned by the policy
                new_state = change_vertex(new_state, action)
                new_state_orig = change_vertex(
                    new_state_orig, positions[action])
                # Update the state
                start = new_state
                start_all = new_state_orig

                _, va, vb = volumes(start_all)

                nnz = start_all.num_edges
                start.x[:, 3] = torch.true_divide(va, nnz)
                start.x[:, 4] = torch.true_divide(vb, nnz)

                time += 1

                # After time_to_sample episods we update the loss
                if i % time_to_sample == 0 or time == len_episode:

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
            if p % print_loss == 0:
                print('worker:', t, 'graph:', p,
                      'reward:', rew_partial)
            rew_partial = 0
            p += 1
    return model

# Deep neural network that models the DRL agent


class Model(torch.nn.Module):
    def __init__(self, units):
        super(Model, self).__init__()

        self.units = units
        self.common_layers = 1
        self.critic_layers = 1
        self.actor_layers = 1
        self.activation = torch.tanh

        self.conv_first = SAGEConv(5, self.units)
        self.conv_common = nn.ModuleList(
            [SAGEConv(self.units, self.units)
             for i in range(self.common_layers)]
        )
        self.conv_actor = nn.ModuleList(
            [SAGEConv(self.units,
                      1 if i == self.actor_layers - 1 else self.units)
             for i in range(self.actor_layers)]
        )
        self.conv_critic = nn.ModuleList(
            [SAGEConv(self.units, self.units)
             for i in range(self.critic_layers)]
        )
        self.final_critic = nn.Linear(self.units, 1)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        do_not_flip = torch.where(x[:, 2] != 0.)

        x = self.activation(self.conv_first(x, edge_index))
        for i in range(self.common_layers):
            x = self.activation(self.conv_common[i](x, edge_index))

        x_actor = x
        for i in range(self.actor_layers):
            x_actor = self.conv_actor[i](x_actor, edge_index)
            if i < self.actor_layers - 1:
                x_actor = self.activation(x_actor)
        x_actor[do_not_flip] = torch.tensor(-np.Inf)
        x_actor = torch.log_softmax(x_actor, dim=0)

        if not self.training:
            return x_actor

        x_critic = x.detach()
        for i in range(self.critic_layers):
            x_critic = self.conv_critic[i](x_critic, edge_index)
            if i < self.critic_layers - 1:
                x_critic = self.activation(x_critic)
        x_critic = self.final_critic(x_critic)
        x_critic = torch.tanh(global_mean_pool(x_critic, batch))
        return x_actor, x_critic

    def forward_c(self, graph, gcsr):
        n = gcsr.shape[0]
        x_actor = torch.zeros([n, 1], dtype=torch.float32)
        libcdrl.forward(
            ctypes.c_int(n),
            ctypes.c_void_p(gcsr.indptr.ctypes.data),
            ctypes.c_void_p(gcsr.indices.ctypes.data),
            ctypes.c_void_p(graph.x.data_ptr()),
            ctypes.c_void_p(x_actor.data_ptr()),
            ctypes.c_void_p(self.conv_first.lin_l.weight.data_ptr()),
            ctypes.c_void_p(self.conv_first.lin_r.weight.data_ptr()),
            ctypes.c_void_p(self.conv_common[0].lin_l.weight.data_ptr()),
            ctypes.c_void_p(self.conv_common[0].lin_r.weight.data_ptr()),
            ctypes.c_void_p(self.conv_actor[0].lin_l.weight.data_ptr()),
            ctypes.c_void_p(self.conv_actor[0].lin_r.weight.data_ptr())
        )
        return x_actor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', default='./temp_edge/', type=str)
    parser.add_argument(
        "--nmin",
        default=200,
        help="Minimum graph size",
        type=int)
    parser.add_argument(
        "--nmax",
        default=5000,
        help="Maximum graph size",
        type=int)
    parser.add_argument(
        "--ntrain",
        default=10000,
        help="Number of training graphs",
        type=int)
    parser.add_argument(
        "--epochs",
        default=1,
        help="Number of training epochs",
        type=int)
    parser.add_argument(
        "--print_rew",
        default=1000,
        help="Steps at which print reward",
        type=int)
    parser.add_argument("--batch", default=8, help="Batch size", type=int)
    parser.add_argument("--hops", default=3, help="Number of hops", type=int)
    parser.add_argument(
        "--workers",
        default=8,
        help="Number of workers",
        type=int)
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
        "--units",
        default=5,
        help="Number of units in conv layers",
        type=int)
    parser.add_argument(
        "--dataset",
        default='delaunay',
        help="Dataset type: delaunay or suitesparse",
        type=str)

    torch.manual_seed(1)
    np.random.seed(2)

    args = parser.parse_args()
    outdir = args.out + '/'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    n_min = args.nmin
    n_max = args.nmax
    n_train = args.ntrain
    episodes = args.epochs
    coeff = args.coeff
    print_loss = args.print_rew

    time_to_sample = args.batch
    hops = args.hops
    n_workers = args.workers
    lr = args.lr
    gamma = args.gamma
    units = args.units
    dataset_type = args.dataset

    # Choose the dataset type according to the parameter 'dataset_type'
    if dataset_type == 'delaunay':
        dataset = delaunay_dataset_with_coarser(n_train, n_min, n_max)
    else:
        dataset = suitesparse_dataset_with_coarser(n_train, n_min, n_max)

    model = Model(units)
    model.share_memory()
    print(model)
    print('Model parameters:',
          sum([w.nelement() for w in model.parameters()]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Start training
    print('Start training')
    t0 = timeit.default_timer()
    processes = []
    for i in range(n_workers):
        p = mp.Process(
            target=training_loop,
            args=(
                i, model, dataset, episodes, gamma, time_to_sample, coeff,
                optimizer, print_loss, hops
            )
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()
    ttrain = timeit.default_timer() - t0
    print('Training took:', ttrain, 'seconds')

    # Saving the model
    if dataset_type == 'delaunay':
        torch.save(model.state_dict(), outdir + 'model_partitioning_delaunay')
    else:
        torch.save(
            model.state_dict(),
            outdir +
            'model_partitioning_suitesparse')
