
import argparse
from pathlib import Path

import networkx as nx
import nxmetis

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, GATConv, GlobalAttention, graclus, avg_pool, global_mean_pool
from torch_geometric.utils import to_networkx, k_hop_subgraph, degree

import numpy as np
from numpy import random

import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy.spatial import Delaunay

#import random_p
import copy
import math
import timeit
import os
from itertools import combinations

import ctypes
libscotch = ctypes.cdll.LoadLibrary('scotch/build/libSCOTCHWrapper.so')


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
    
    adj_sparse = nx.to_scipy_sparse_array(g, format='coo')
    row = adj_sparse.row
    col = adj_sparse.col

    one_hot = []
    for i in range(g.number_of_nodes()):
        one_hot.append([1., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(one_hot), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)
    
    return graph_torch

# Build a pytorch geometric graph with features [1,0] form a networkx graph
    
def torch_from_graph(g):

    adj_sparse = nx.to_scipy_sparse_array(g, format='coo')
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


def k_hop_graph_cut(graph, k, g, va, vb):
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

# Full valuation of the DRL model


def ac_eval_coarse_full_drl(ac, graph, k, ac2):
    g = graph.clone()
    info = []
    edge_info = []
    while g.num_nodes > 100:
        edge_info.append(g.edge_index)
        cluster = graclus(g.edge_index)
        info.append(cluster)
        g1 = avg_pool(
            cluster,
            Batch(
                batch=g.batch,
                x=g.x,
                edge_index=g.edge_index))
        g = g1

    gnx = to_networkx(g, to_undirected=True)
    g = torch_from_graph(gnx)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    g = ac_eval(ac2, g, 0.01)

    while len(info) > 0:
        cluster = info.pop()
        _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
        g.x = g.x[inverse]
        g.edge_index = edge_info.pop()
        _, volA, volB = volumes(g)
        gnx = to_networkx(g, to_undirected=True)
        g = ac_eval_refine(ac, g, k, gnx, volA, volB)
    return g

# Full valuation of the DRL model repeated for trials number of times.
# Then the best partition is returned


def ac_eval_coarse_full_trials_drl(ac, graph, k, trials, ac2):
    graph_test = graph.clone()
    gg = ac_eval_coarse_full_drl(ac, graph_test, k, ac2)
    ncut = normalized_cut(gg)
    for j in range(1, trials):
        gg1 = ac_eval_coarse_full_drl(ac, graph_test, k, ac2)
        if normalized_cut(gg1) < ncut:
            ncut = normalized_cut(gg1)
            gg = gg1

    return gg

# Full valuation of the DRL_METIS model repeated for trials number of
# times. Then the best partition is returned


def ac_eval_coarse_full_trials(ac, graph, k, trials):
    graph_test = graph.clone()
    gg = ac_eval_coarse_full(ac, graph_test, k)
    ncut = normalized_cut(gg)
    for j in range(1, trials):
        gg1 = ac_eval_coarse_full(ac, graph_test, k)
        if normalized_cut(gg1) < ncut:
            ncut = normalized_cut(gg1)
            gg = gg1

    return gg

# Full valuation of the DRL_METIS model


def ac_eval_coarse_full(ac, graph, k):
    g = graph.clone()
    info = []
    edge_info = []
    while g.num_nodes > 100:
        edge_info.append(g.edge_index)
        cluster = graclus(g.edge_index)
        info.append(cluster)
        g1 = avg_pool(
            cluster,
            Batch(
                batch=g.batch,
                x=g.x,
                edge_index=g.edge_index))
        g = g1

    gnx = to_networkx(g, to_undirected=True)
    g = partition_metis(g, gnx)

    while len(info) > 0:
        cluster = info.pop()
        _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
        g.x = g.x[inverse]
        g.edge_index = edge_info.pop()
        _, volA, volB = volumes(g)
        gnx = to_networkx(g, to_undirected=True)
        g = ac_eval_refine(ac, g, k, gnx, volA, volB)

    return g

# Partitioning of a pytorch geometric graph obtained with METIS


def partition_metis(graph, graph_nx):
    obj, parts = nxmetis.partition(graph_nx, 2)
    #mparts = np.array(parts)
    graph.x[parts[0]] = torch.tensor([1., 0.])
    graph.x[parts[1]] = torch.tensor([0., 1.])

    return graph

# Refining the cut on the subgraph around the cut


def ac_eval_refine(ac, graph_t, k, gnx, volA, volB):
    graph = graph_t.clone()
    g0 = graph_t.clone()
    data = k_hop_graph_cut(graph, k, gnx, volA, volB)
    graph_cut, positions = data[0], data[1]

    len_episod = int(cut(graph))

    peak_reward = 0
    peak_time = 0
    total_reward = 0
    actions = []

    e = torch.ones(graph_cut.num_nodes, 1)
    nnz = graph.num_edges
    cut_sub = len_episod
    for i in range(len_episod):
        with torch.no_grad():
            policy = ac(graph_cut)
        probs = policy.view(-1).clone().detach().numpy()
        flip = np.argmax(probs)

        dv = gnx.degree[positions[flip].item()]
        old_nc = cut_sub * (torch.true_divide(1, volA) +
                            torch.true_divide(1, volB))
        if graph_cut.x[flip, 0] == 1.:
            volA = volA - dv
            volB = volB + dv
        else:
            volA = volA + dv
            volB = volB - dv
        new_nc, cut_sub = update_nc(
            graph, gnx, cut_sub, positions[flip].item(), volA, volB)
        total_reward += (old_nc - new_nc).item()

        actions.append(flip)

        change_vertex(graph_cut, flip)
        change_vertex(graph, positions[flip])

        graph_cut.x[:, 3] = torch.true_divide(volA, nnz)
        graph_cut.x[:, 4] = torch.true_divide(volB, nnz)

        if i >= 1 and actions[-1] == actions[-2]:
            break
        if total_reward > peak_reward:
            peak_reward = total_reward
            peak_time = i + 1

    for t in range(peak_time):
        g0 = change_vertex(g0, positions[actions[t]])

    return g0

# Compute the update for the normalized cut


def update_nc(graph, gnx, cut_total, v1, va, vb):
    c_v1 = 0
    for v in gnx[v1]:
        if graph.x[v, 0] != graph.x[v1, 0]:
            c_v1 += 1
        else:
            c_v1 -= 1
    cut_new = cut_total - c_v1
    return cut_new * (torch.true_divide(1, va) +
                      torch.true_divide(1, vb)), cut_new

# Evaluation of the DRL model on the coarsest graph


def ac_eval(ac, graph, perc):
    graph_test = graph.clone()
    error_bal = math.ceil(graph_test.num_nodes * perc)
    cuts = []
    nodes = []
    # Run the episod
    for i in range(int(graph_test.num_nodes / 2 - 1 + error_bal)):
        policy, _ = ac(graph_test)
        policy = policy.view(-1).detach().numpy()
        flip = random.choice(torch.arange(0, graph_test.num_nodes), p=policy)
        graph_test = change_vertex(graph_test, flip)
        if i >= int(graph_test.num_nodes / 2 - 1 - error_bal):
            cuts.append(cut(graph_test))
            nodes.append(flip)
    if len(cuts) > 0:
        stops = np.argwhere(cuts == np.min(cuts))
        stops = stops.reshape((stops.shape[0],))
        if len(stops) == 1:
            graph_test.x[nodes[stops[0] + 1:]] = torch.tensor([1., 0.])
        else:
            diff = [np.abs(i - int(len(stops) / 2 - 1)) for i in stops]
            min_dist = np.argwhere(diff == np.min(diff))
            min_dist = min_dist.reshape((min_dist.shape[0],))
            stop = np.random.choice(stops[min_dist])
            graph_test.x[nodes[stop + 1:]] = torch.tensor([1., 0.])

    return graph_test

# Partitioning provided by SCOTCH


def scotch_partition(g):
    gnx = to_networkx(g, to_undirected=True)
    a = nx.to_scipy_sparse_array(gnx, format="csr", dtype=np.float32)
    n = g.num_nodes
    part = np.zeros(n, dtype=np.int32)
    libscotch.WRAPPER_SCOTCH_graphPart(
        ctypes.c_int(n),
        ctypes.c_void_p(a.indptr.ctypes.data),
        ctypes.c_void_p(a.indices.ctypes.data),
        ctypes.c_void_p(part.ctypes.data)
    )
    g.x[np.where(part == 0)] = torch.tensor([1., 0.])
    g.x[np.where(part == 1)] = torch.tensor([0., 1.])
    return g

# Deep neural network for the DRL agent


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
        default=100,
        help="Minimum graph size",
        type=int)
    parser.add_argument(
        "--nmax",
        default=10000,
        help="Maximum graph size",
        type=int)
    parser.add_argument(
        "--ntest",
        default=1000,
        help="Number of test graphs",
        type=int)
    parser.add_argument("--hops", default=3, help="Number of hops", type=int)
    parser.add_argument(
        "--units",
        default=5,
        help="Number of units in conv layers",
        type=int)
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
    parser.add_argument(
        "--attempts",
        default=3,
        help="Number of attempts in the DRL",
        type=int)
    parser.add_argument(
        "--dataset",
        default='delaunay',
        help="Dataset type: delaunay, suitesparse, graded l, hole3, hole6",
        type=str)

    torch.manual_seed(1)
    np.random.seed(2)

    args = parser.parse_args()
    outdir = args.out + '/'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    n_min = args.nmin
    n_max = args.nmax
    n_test = args.ntest
    hops = args.hops
    units = args.units
    trials = args.attempts
    hid_conv = args.units_conv
    hid_lin = args.units_dense
    dataset_type = args.dataset

    # Deep neural network for the DRL agent on the coarsest graph
    class ModelCoarsest(torch.nn.Module):
        def __init__(self):
            super(ModelCoarsest, self).__init__()
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

    # Choose the model to load according to the parameter 'dataset_type'
    model = Model(units)
    if dataset_type == 'suitesparse':
        model.load_state_dict(
            torch.load('./temp_edge/model_partitioning_suitesparse'))
    else:
        model.load_state_dict(
            torch.load('./temp_edge/model_partitioning_delaunay'))

    model_coarsest = ModelCoarsest()
    model_coarsest.load_state_dict(torch.load('./temp_edge/model_coarsest'))
    model.eval()
    model_coarsest.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model_coarsest.parameters():
        p.requires_grad = False

    print('Models loaded\n')
    list_picked = []
    i = 0
    while i < n_test:
        # Choose the dataset type according to the parameter 'dataset_type'
        if dataset_type == 'delaunay':
            n_nodes = np.random.choice(np.arange(n_min, n_max))
            g = random_delaunay_graph(n_nodes)
            g.batch = torch.zeros(g.num_nodes)
            i += 1
        else:
            if len(list_picked) >= len(
                os.listdir(
                    os.path.expanduser(
                        'drl-graph-partitioning/' +
                        str(dataset_type) +
                        '/'))):
                break
            graph = random.choice(
                os.listdir(
                    os.path.expanduser(
                        'drl-graph-partitioning/' +
                        str(dataset_type) +
                        '/')))
            if str(graph) not in list_picked:
                list_picked.append(str(graph))
                matrix_sparse = mmread(
                    os.path.expanduser(
                        'drl-graph-partitioning/' +
                        str(dataset_type) +
                        '/' +
                        str(graph)))
                gnx = nx.from_scipy_sparse_array(matrix_sparse)
                if nx.number_connected_components(gnx) == 1 and gnx.number_of_nodes(
                ) > n_min and gnx.number_of_nodes() < n_max:
                    g = torch_from_sparse(matrix_sparse)
                    g.batch = torch.zeros(g.num_nodes)
                    i += 1
                else:
                    continue
            else:
                continue
        print('Graph:', i, '  Vertices:', g.num_nodes, '  Edges:', g.num_edges)

        gnx = to_networkx(g, to_undirected=True)

        # Partitioning with DRL
        graph_p = ac_eval_coarse_full_trials_drl(
            model, g, hops, trials, model_coarsest)
        cdrl, a, b = volumes(graph_p)

        # Partitioning with DRL_METIS
        graph_p1 = ac_eval_coarse_full_trials(model, g, hops, trials)
        cdrl_m, a1, b1 = volumes(graph_p1)

        # Partitioning with METIS
        cut_met, parts = nxmetis.partition(gnx, 2)
        #mparts = np.array(parts)
        a_m = sum(gnx.degree(i) for i in parts[0])
        b_m = sum(gnx.degree(i) for i in parts[1])

        # Partitioning with SCOTCH
        gscotch = scotch_partition(g)
        csc, a_s, b_s = volumes(gscotch)

        # Print the results
        print(
            'NC:  DRL:',
            np.round(
                normalized_cut(graph_p),
                5),
            '  DRL_METIS:',
            np.round(
                normalized_cut(graph_p1),
                5),
            '  METIS:',
            np.round(
                cut_met * (
                    1 / a_m + 1 / b_m),
                5),
            '  SCOTCH:',
            np.round(
                normalized_cut(gscotch),
                5))
        print('Volumes:  DRL:', (a, b), '  DRL_METIS:', (a1, b1),
              '  METIS:', (a_m, b_m), '  SCOTCH:', (a_s, b_s))
        print(
            'Cut:  DRL:',
            int(cdrl),
            '  DRL_METIS:',
            int(cdrl_m),
            '  METIS:',
            int(cut_met),
            '  SCOTCH:',
            int(csc))
        print('')
    print('Done')
