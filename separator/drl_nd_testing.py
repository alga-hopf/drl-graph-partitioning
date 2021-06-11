
import argparse
from pathlib import Path

import networkx as nx
import nxmetis

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, graclus, avg_pool, global_mean_pool
from torch_geometric.utils import to_networkx, k_hop_subgraph, degree, subgraph

import numpy as np
from numpy import random

import scipy
from scipy.sparse import coo_matrix, rand, identity, csc_matrix
from scipy.io import mmread
from scipy.spatial import Delaunay
from scipy.sparse.linalg import splu

import copy
import os
from itertools import combinations

import ctypes
libamd = ctypes.cdll.LoadLibrary('amd/build/libAMDWrapper.so')
libscotch = ctypes.cdll.LoadLibrary('scotch/build/libSCOTCHWrapper.so')

# Full evaluation of the DRL model


def ac_eval_coarse_full(ac, graph, k):
    g = graph.clone()
    info = []
    edge_info = []
    while g.num_nodes > 100:
        edge_info.append(g.edge_index)
        cluster = graclus(g.edge_index, num_nodes=g.num_nodes)
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
        gnx = to_networkx(g, to_undirected=True)
        va, vb = volumes(g)
        g = ac_eval_refine(ac, g, k, gnx, va, vb)
    return g

# Refining the cut on the subgraph around the separator


def ac_eval_refine(ac, graph_test, k, gnx, va, vb, perc=0.05):
    graph = graph_test.clone()
    g0 = graph_test.clone()
    n = separator(graph)
    data = k_hop_graph_cut(graph, k)
    graph_cut, positions = data[0], data[1]
    gnx_sub = to_networkx(graph_cut, to_undirected=True)
    i = 0

    peak_reward = 0
    peak_time = 0
    total_reward = 0
    actions = []

    e = torch.ones(graph_cut.num_nodes, 1)
    nnz = graph.num_nodes
    sep = int(n)
    nodes_separator = torch.where((graph_cut.x[:, 2] == torch.tensor(1.)))[0]

    for i in range(int(2 * n)):
        with torch.no_grad():
            policy = ac(graph_cut)

        probs = policy.view(-1).clone().detach().numpy()

        flip = np.argmax(probs)

        actions.append(flip)
        old_sep = sep * (1 / va + 1 / vb)

        graph_cut, a, b, s = change_vertex(graph_cut, flip, gnx_sub, va, vb)
        graph, _, _, _ = change_vertex(
            graph, positions[flip].item(), gnx, va, vb)

        if a == 1:
            va += 1
            sep -= 1
        elif a == -1:
            va -= 1
            sep += 1
        elif b == 1:
            vb += 1
            sep -= 1
        elif b == -1:
            vb -= 1
            sep += 1

        total_reward += old_sep - sep * (1 / va + 1 / vb)

        graph_cut.x[:, 5] = torch.true_divide(va, nnz)
        graph_cut.x[:, 6] = torch.true_divide(vb, nnz)

        nodes_separator = torch.where(
            (graph_cut.x[:, 2] == torch.tensor(1.)))[0]
        graph_cut = remove_update(graph_cut, gnx_sub, nodes_separator)

        if i > 1 and actions[-1] == actions[-2]:
            break
        if total_reward > peak_reward:
            peak_reward = total_reward
            peak_time = i + 1

    for t in range(peak_time):
        g0, _, _, _ = change_vertex(
            g0, positions[actions[t]].item(), gnx, va, vb)

    return g0

# Update the nodes that are necessary to get a minimal separator


def remove_update(gr, gnx, sep):
    graph = gr.clone()
    for ii in sep:
        i = ii.item()
        flagA, flagB = 0, 0
        for v in gnx[i]:
            if flagA == 1 and flagB == 1:
                break
            if graph.x[v, 0] == torch.tensor(1.):
                flagA = 1
            elif graph.x[v, 1] == torch.tensor(1.):
                flagB = 1
        if flagA == 1 and flagB == 1:
            graph.x[i, 4] = torch.tensor(1.)
        else:
            graph.x[i, 4] = torch.tensor(0.)
    return graph

# Full valuation of the DRL model repeated for trials number of times.
# Then the best separator is returned


def ac_eval_coarse_full_trials(ac, graph, k, trials):
    graph_test = graph.clone()
    gg = ac_eval_coarse_full(ac, graph_test, k)
    ncut = normalized_separator(gg)
    for j in range(1, trials):
        gg1 = ac_eval_coarse_full(ac, graph_test, k)
        if normalized_separator(gg1) < ncut:
            ncut = normalized_separator(gg1)
            gg = gg1

    return gg

# Change the feature of the selected vertex v


def change_vertex(g, v, gnx, va, vb):
    a, b, s = 0, 0, 0
    if g.x[v, 2] == 0.:
        if g.x[v, 0] == 1.:
            a, s = -1, 1
        else:
            b, s = -1, 1
        # node v is in A or B, add it to the separator
        g.x[v, :3] = torch.tensor([0., 0., 1.])

        return g, a, b, s
    # node v is in the separator

    for vj in gnx[v]:
        if g.x[vj, 0] == 1.:
            # node v is in the separator and connected to A, so add it
            # to A
            g.x[v, :3] = torch.tensor([1., 0., 0.])
            a, s = 1, -1
            return g, a, b, s
        if g.x[vj, 1] == 1.:
            # node v is in the separator and connected to B, so add it
            # to B
            g.x[v, :3] = torch.tensor([0., 1., 0.])
            b, s = 1, -1
            return g, a, b, s
    # node v is in the separator, but is not connected to A or B.  Add
    # node v to A if the volume of A is less (or equal) to that of B,
    # or to B if the volume of B is less than that of A.
    if va <= vb:
        g.x[v, :3] = torch.tensor([1., 0., 0.])
        s, a = -1, 1
    else:
        g.x[v, :3] = torch.tensor([0., 1., 0.])
        s, b = -1, 1
    return g, a, b, s

# Build a pytorch geometric graph with features [1,0,0] form a networkx graph


def torch_from_graph(graph):
    adj_sparse = nx.to_scipy_sparse_matrix(graph, format='coo')
    row = adj_sparse.row
    col = adj_sparse.col

    one_hot = []
    for i in range(graph.number_of_nodes()):
        one_hot.append([1., 0., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(one_hot), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    return graph_torch

# Build a pytorch geometric graph with features [1,0] form a sparse matrix


def torch_from_sparse(adj_sparse):

    row = adj_sparse.row
    col = adj_sparse.col

    features = []
    for i in range(adj_sparse.shape[0]):
        features.append([1., 0., 0.])

    edges = torch.tensor([row, col], dtype=torch.long)
    nodes = torch.tensor(np.array(features), dtype=torch.float)
    graph_torch = Data(x=nodes, edge_index=edges)

    return graph_torch

# Pytorch geometric Delaunay mesh with n random points in the unit square


def random_delaunay_graph(n):
    points = np.random.random_sample((n, 2))
    g = graph_delaunay_from_points(points)
    return torch_from_graph(g)

# Networkx Delaunay mesh with n random points in the unit square


def graph_delaunay_from_points(points):
    mesh = Delaunay(points, qhull_options="QJ")
    mesh_simp = mesh.simplices
    edges = []
    for i in range(len(mesh_simp)):
        edges += combinations(mesh_simp[i], 2)
    e = list(set(edges))
    return nx.Graph(e)

# Number of vertices in the separator


def separator(graph):
    sep = torch.where((graph.x == torch.tensor(
        [0., 0., 1.])).all(axis=-1))[0].shape[0]
    return sep

# Normalized separator


def normalized_separator(graph):
    da, db = volumes(graph)
    sep = torch.where((graph.x == torch.tensor(
        [0., 0., 1.])).all(axis=-1))[0].shape[0]
    if da == 0 or db == 0:
        return 10
    else:
        return sep * (1 / da + 1 / db)

# Normalized separator for METIS


def vertex_sep_metis(graph, gnx):
    sep, nodes1, nodes2 = nxmetis.vertex_separator(gnx)
    da = len(nodes1)
    db = len(nodes2)
    return len(sep) * (1 / da + 1 / db)

# Subgraph around the separator


def k_hop_graph_cut(graph, k):
    nei = torch.where((graph.x[:, 2] == torch.tensor(1.)))[0]
    data_cut = k_hop_subgraph(
        nei,
        k,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes)
    data_small = k_hop_subgraph(
        nei,
        k - 1,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes)
    nodes_boundary = list(
        set(data_cut[0].numpy()).difference(data_small[0].numpy()))
    boundary_features = torch.tensor([1. if i.item(
    ) in nodes_boundary else 0. for i in data_cut[0]]).reshape(data_cut[0].shape[0], 1)
    remove_f = []
    for j in range(len(data_cut[0])):
        if graph.x[data_cut[0][j]][2] == torch.tensor(1.):
            neighbors, _, _, _ = k_hop_subgraph(
                [data_cut[0][j]], 1, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes)
            flagA, flagB = 0, 0
            for w in neighbors:
                if graph.x[w][0] == torch.tensor(1.):
                    flagA = 1
                elif graph.x[w][1] == torch.tensor(1.):
                    flagB = 1
            if flagA == 1 and flagB == 1:
                remove_f.append(1.)
            else:
                remove_f.append(0.)
        else:
            remove_f.append(0.)
    remove_features = torch.tensor(remove_f).reshape(len(remove_f), 1)
    va, vb = volumes(graph)
    e = torch.ones(data_cut[0].shape[0], 1)
    nnz = graph.num_nodes
    features = torch.cat((graph.x[data_cut[0]],
                          boundary_features,
                          remove_features,
                          torch.true_divide(va,
                                            nnz) * e,
                          torch.true_divide(vb,
                                            nnz) * e),
                         1)
    g_red = Batch(
        batch=torch.zeros(
            data_cut[0].shape[0],
            dtype=torch.long),
        x=features,
        edge_index=data_cut[1])
    return g_red, data_cut[0]

# Cardinalities of the partitions A and B


def volumes(graph):
    ab = torch.sum(graph.x, dim=0)
    return ab[0].item(), ab[1].item()

# Coarsen a pytorch geometric graph, then find the separator with METIS
# and interpolate it back


def partition_metis_refine(graph):
    cluster = graclus(graph.edge_index, num_nodes=graph.num_nodes)
    coarse_graph = avg_pool(
        cluster,
        Batch(
            batch=graph.batch,
            x=graph.x,
            edge_index=graph.edge_index))
    coarse_graph_nx = to_networkx(coarse_graph, to_undirected=True)
    sep, A, B = nxmetis.vertex_separator(coarse_graph_nx)
    coarse_graph.x[sep] = torch.tensor([0., 0., 1.])
    coarse_graph.x[A] = torch.tensor([1., 0., 0.])
    coarse_graph.x[B] = torch.tensor([0., 1., 0.])
    _, inverse = torch.unique(cluster, sorted=True, return_inverse=True)
    graph.x = coarse_graph.x[inverse]
    return graph

# Separator of a pytorch geometric graph obtained with METIS


def partition_metis(coarse_graph, coarse_graph_nx):
    sep, A, B = nxmetis.vertex_separator(coarse_graph_nx)
    coarse_graph.x[sep] = torch.tensor([0., 0., 1.])
    coarse_graph.x[A] = torch.tensor([1., 0., 0.])
    coarse_graph.x[B] = torch.tensor([0., 1., 0.])
    return coarse_graph

# Matrix ordering provided by SCOTCH


def scotch_ordering(g):
    gnx = to_networkx(g, to_undirected=True)
    a = nx.to_scipy_sparse_matrix(gnx, format="csr", dtype=np.float32)
    n = g.num_nodes
    perm = np.zeros(n, dtype=np.int32)
    libscotch.WRAPPER_SCOTCH_graphOrder(
        ctypes.c_int(n),
        ctypes.c_void_p(a.indptr.ctypes.data),
        ctypes.c_void_p(a.indices.ctypes.data),
        ctypes.c_void_p(perm.ctypes.data)
    )
    return perm.tolist()

# Matrix ordering provided by COLAMD


def amd_ordering(g):
    gnx = to_networkx(g, to_undirected=True)
    n = g.num_nodes
    a = nx.to_scipy_sparse_matrix(gnx, format="csr", dtype=np.float32)
    a += identity(n)
    perm = np.zeros(n, dtype=np.int32)
    iperm = np.zeros(n, dtype=np.int32)
    libamd.WRAPPER_amd(
        ctypes.c_int(n),
        ctypes.c_void_p(a.indptr.ctypes.data),
        ctypes.c_void_p(a.indices.ctypes.data),
        ctypes.c_void_p(perm.ctypes.data),
        ctypes.c_void_p(iperm.ctypes.data)
    )
    return perm.tolist()

# Nested dissection ordering with the DRL algorithm


def drl_nested_dissection(graph, nmin, hops, model, trials, lvl=0):
    g_stack = [graph]
    i_stack = [[i for i in range(graph.num_nodes)]]
    perm = []
    i = 0
    while g_stack:
        g = g_stack.pop()
        idx = i_stack.pop()
        if g.num_nodes < nmin:
            if g.num_nodes > 0:
                p = amd_ordering(g)
                perm = [idx[i] for i in p] + perm
        else:
            g = ac_eval_coarse_full_trials(model, g, hops, trials)
            ia = torch.where(g.x[:, 0] == 1.)[0].tolist()
            ib = torch.where(g.x[:, 1] == 1.)[0].tolist()
            isep = torch.where(g.x[:, 2] == 1.)[0].tolist()
            ga_data = subgraph(
                ia, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes
            )[0]
            gb_data = subgraph(
                ib, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes
            )[0]
            ga = Batch(
                batch=torch.zeros(len(ia), 1),
                x=torch.zeros(len(ia), 3), edge_index=ga_data
            )
            gb = Batch(
                batch=torch.zeros(len(ib), 1),
                x=torch.zeros(len(ib), 3), edge_index=gb_data
            )
            g_stack.append(ga)
            i_stack.append([idx[i] for i in ia])
            g_stack.append(gb)
            i_stack.append([idx[i] for i in ib])
            perm = [idx[i] for i in isep] + perm
        i += 1
    return perm

# Nested dissection ordering implemented with METIS


def metis_nested_dissection(graph, nmin):
    g_stack = [graph]
    i_stack = [[i for i in range(graph.num_nodes)]]
    perm = []
    i = 0
    while g_stack:
        g = g_stack.pop()
        idx = i_stack.pop()
        if g.num_nodes < nmin:
            if g.num_nodes > 0:
                p = amd_ordering(g)
                perm = [idx[i] for i in p] + perm
        else:
            isep, ia, ib = nxmetis.vertex_separator(
                to_networkx(g, to_undirected=True))
            ga_data = subgraph(
                ia, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes
            )[0]
            gb_data = subgraph(
                ib, g.edge_index, relabel_nodes=True, num_nodes=g.num_nodes
            )[0]
            ga = Batch(
                batch=torch.zeros(len(ia), 1),
                x=torch.zeros(len(ia), 3), edge_index=ga_data
            )
            gb = Batch(
                batch=torch.zeros(len(ib), 1),
                x=torch.zeros(len(ib), 3), edge_index=gb_data
            )
            g_stack.append(ga)
            i_stack.append([idx[i] for i in ia])
            g_stack.append(gb)
            i_stack.append([idx[i] for i in ib])
            perm = [idx[i] for i in isep] + perm
        i += 1
    return perm

# Deep neural network for the DRL agent


class Model(torch.nn.Module):
    def __init__(self, units):
        super(Model, self).__init__()

        self.units = units
        self.common_layers = 1
        self.critic_layers = 1
        self.actor_layers = 1
        self.activation = torch.tanh

        self.conv_first = SAGEConv(7, self.units)
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

        do_not_flip = torch.where(x[:, 3] != 0.)
        do_not_flip_2 = torch.where(x[:, 4] != 0.)

        x = self.activation(self.conv_first(x, edge_index))
        for i in range(self.common_layers):
            x = self.activation(self.conv_common[i](x, edge_index))

        x_actor = x
        for i in range(self.actor_layers):
            x_actor = self.conv_actor[i](x_actor, edge_index)
            if i < self.actor_layers - 1:
                x_actor = self.activation(x_actor)
        x_actor[do_not_flip] = torch.tensor(-np.Inf)
        x_actor[do_not_flip_2] = torch.tensor(-np.Inf)
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
        default=50000,
        help="Maximum graph size",
        type=int)
    parser.add_argument(
        "--ntest",
        default=1000,
        help="Number of testing graphs",
        type=int)
    parser.add_argument("--hops", default=3, help="Number of hops", type=int)
    parser.add_argument(
        "--units",
        default=7,
        help="Number of units in conv layers",
        type=int)
    parser.add_argument(
        "--attempts",
        default=3,
        help="Number of attempt in the DRL",
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
    dataset_type = args.dataset

    model = Model(units)
    if dataset_type == 'suitesparse':
        model.load_state_dict(
            torch.load('./temp_edge/model_separator_suitesparse'))
    else:
        model.load_state_dict(
            torch.load('./temp_edge/model_separator_delaunay'))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print('Model loaded\n')

    nmin_nd = 100

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
                gnx = nx.from_scipy_sparse_matrix(matrix_sparse)
                if nx.number_connected_components(gnx) == 1 and gnx.number_of_nodes(
                ) > n_min and gnx.number_of_nodes() < n_max:
                    g = torch_from_sparse(matrix_sparse)
                    g.weight = torch.tensor([1] * g.num_edges)
                    g.batch = torch.zeros(g.num_nodes)
                    i += 1
                else:
                    continue
            else:
                continue
        print('Graph:', i, '  Vertices:', g.num_nodes, '  Edges:', g.num_edges)

        gnx = to_networkx(g, to_undirected=True)
        a = nx.to_scipy_sparse_matrix(
            gnx, format='csc') + 10 * identity(g.num_nodes)

        # Compute the number of non-zero (nnz) in elements in the LU
        # factorization with DRL
        # Sometimes METIS may fail in computing the vertex separator on the
        # coarsest graph, producing an empty partition that affects the
        # computations on the finer interpolation levels
        try:
            p = drl_nested_dissection(g, nmin_nd, hops, model, trials, lvl=0)
        except ZeroDivisionError:
            continue
        aperm = a[:, p][p, :]
        lu = splu(aperm, permc_spec='NATURAL')
        nnz_drl = lu.L.count_nonzero() + lu.U.count_nonzero()

        # Compute the number of non-zero (nnz) elements in the LU factorization
        # with nested dissection with METIS
        p = metis_nested_dissection(g, nmin_nd)
        aperm = a[:, p][p, :]
        lu = splu(aperm, permc_spec='NATURAL')
        nnz_nd_metis = lu.L.count_nonzero() + lu.U.count_nonzero()

        # Compute the number of non-zero (nnz) elements in the LU factorization
        # with COLAMD (this is the default ordering for superlu)
        lu = splu(a, permc_spec='COLAMD')
        nnz_colamd = lu.L.count_nonzero() + lu.U.count_nonzero()

# Compute the number of non-zero (nnz) elements in the LU factorization
# with the built-in nested dissection ordering with METIS
        p = nxmetis.node_nested_dissection(gnx)
        aperm = a[:, p][p, :]
        lu = splu(aperm, permc_spec='NATURAL')
        nnz_metis = lu.L.count_nonzero() + lu.U.count_nonzero()

# Compute the number of non-zero (nnz) elements in the LU factorization
# with SCOTCH
        p = scotch_ordering(g)
        aperm = a[:, p][p, :]
        lu = splu(aperm, permc_spec='NATURAL')
        nnz_scotch = lu.L.count_nonzero() + lu.U.count_nonzero()
        print(
            'NNZ: DRL:',
            nnz_drl,
            '  ND_METIS:',
            nnz_nd_metis,
            '  AMD:,',
            nnz_colamd,
            '  METIS:',
            nnz_metis,
            '  SCOTCH:',
            nnz_scotch)
        print('')
    print('Done')
