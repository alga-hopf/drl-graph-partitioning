# Graph Partitioning and Sparse Matrix Ordering using Reinforcement Learning and Graph Neural Networks

Deep reinforcement learning models to find a minimum normalized cut partitioning and a minimal vertex separator from https://arxiv.org/abs/2104.03546.

## Minimum normalized cut

Given a graph G=(V,E), the goal is to find a partition of the set of vertices V such that its normalized cut is minimized. This is known to be an NP-complete problem, hence we look for approximate solutions to it. In order to do that, we train a Distributed Advantage Actor-Critic (DA2C) agent to refine the partitions obtained by interpolating back the partition on a coarser representation of the graph. More precisely, the initial graph is coarsened until it has a small number of nodes, then the coarsest graph is partitioned and the partition is interpolated back one level, where it is further refined. This process is applied recursively up to the initial graph. To partition the coarsest graph one can use the software METIS (through the NetworkX-METIS wrapper) or train another deep reinforcement learning agent to partition it. Both agents are implemented by two-headed deep neural networks containing graph convolutional layers.

The details about the constructions and the trainings can be found in Section 2 and Section 3 of the paper. The codes for the training (refining and partitioning the coarsest graph) and for the evaluation are in the ``partitioning`` folder. 

### Training the DRL agent to partition the coarsest graph
Run ``drl_partitioning_coarsest_train.py`` with the following arguments
- ``out``: output folder to save the weights of the trained neural network (default: ``./temp_edge``)
- ``nmin``: minimum graph size (default: 50)
- ``nmax``: maximum graph size (default: 100)
- ``ntrain``: number of training graphs (default: 10000)
- ``print_rew``: steps to take before printing the reward (default: 1000)
- ``lr``: learning rate (default: 0.001)
- ``gamma``: discount factor (default: 0.9)
- ``coeff``: critic loss coefficient (default: 0.1)
- ``units_conv``: units for the 4 convolutional layers (default: 30, 30, 30, 30)
- ``units_dense``: units for the 3 linear layers (default: 30, 30, 20)

After the training the weights are saved in ``out`` with the name ``model_coarsest``.

### Training the DRL agent to refine a given partitioned graph
Run ``drl_partitioning_train.py`` with the following arguments
- ``out``: output folder to save the weights of the trained neural network (default: ``./temp_edge``)
- ``nmin``: minimum graph size (default: 200)
- ``nmax``: maximum graph size (default: 5000)
- ``ntrain``: number of training graphs (default: 10000)
- ``epochs``: number of epochs (default: 1)
- ``print_rew``: steps to take before printing the reward (default: 1000)
- ``batch``: steps to take before updating the loss function (default: 8)
- ``hops``: number of hops (default: 3)
- ``workers``: number of workers (default: 8)
- ``lr``: learning rate (default: 0.001)
- ``gamma``: discount factor (default: 0.9)
- ``coeff``: critic loss coefficient (default: 0.1)
- ``units``: number of units in the graph convolutional layers (default: 5)
- ``dataset``: dataset type to choose between ``'delaunay'`` and ``'suitesparse'`` (default: ``'delaunay'``). With the first choice, random Delaunay graphs in the unit square are generated before the training. With the second choice, the user needs to download the matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) in the Matrix Market format and put the ``.mtx`` files in the folder ``drl-graph-partitioning/suitesparse_train``. In the paper we focused on matrices coming from 2D/3D discretizations.

After the training the weights are saved in ``out`` with the name ``model_partitioning_delaunay``. 

### Testing the DRL agent on several datasets
Run ``drl_partitioning_test.py`` with the following arguments
- ``out``: output folder to save the weights of the trained neural network (default: ``./temp_edge``)
- ``nmin``: minimum graph size (default: 100)
- ``nmax``: maximum graph size (default: 10000)
- ``ntest``: number of testing graphs (default: 1000)
- ``hops``: : number of hops (default: 3)
- ``units``: number of units in the graph convolutional layers in the loaded refining DNN (default: 5)
- ``units_conv``: units for the 4 convolutional layers in the loaded DNN for the coarsest graph (default: 30, 30, 30, 30)
- ``units_dense``: units for the 3 linear layers in the DNN for the coarsest graph (default: 30, 30, 20)
- ``attempts``: number of attempts to make (default: 3)
- ``dataset``: dataset type to choose among ``'delaunay'``, ``'suitesparse'``, and the Finite Elements triangulations ``graded_l``, ``hole3``, ``hole6`` (default: ``'delaunay'``). With the first choice, random Delaunay graphs in the unit square are generated before the evaluation. With the second choice, the user needs to download the matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) in the Matrix Market format and put the ``.mtx.`` files in the folder ``drl-graph-partitioning/suitesparse``. For the Finite Elements triangulations, the user can download the matrices from [here](https://portal.nersc.gov/project/sparse/strumpack/fe_triangulations.tar.xz) and put the 3 folders in ``drl-graph-partitioning/``.

Make sure that the arguments ``units``, ``units_conv`` and ``units_dense`` are the same used in the training phases.
For each graph in the dataset the following are returned: normalized cut, corresponding volumes and cut computed with DRL, DRL_METIS, METIS and SCOTCH.

## Minimal vertex separator

Given a graph G=(V,E), the goal is to find a minimal vertex separator such that the corresponding normalized separator is minimized. Also in this case we look for approximate solutions to it. In order to do that, we train a Distributed Advantage Actor-Critic (DA2C) agent to refine the vertex separator obtained by interpolating back the partition on a coarser representation of the graph. More precisely, the initial graph is coarsened until it has a small number of nodes, then a minimal vertex separator is computed on the the coarsest graph and the it is interpolated back one level, where it is further refined. This process is applied recursively up to the initial graph. To find the minimal vertex separator the coarsest graph one can use the software METIS (through the NetworkX-METIS wrapper). The agent is implemented by two-headed deep neural network containing graph convolutional layers.

The details about the constructions and the trainings can be found in Section 4 the paper, while Section 5 contains an application of the above model to the Nested Dissection Sparse Matrix Ordering. The codes for the training, for the evaluation and for the nested dissection ordering are in the ``separator`` folder. 

### Training the DRL agent to refine a given vertex separator partition
Run ``drl_separator_train.py`` with the following arguments
- ``out``: output folder to save the weights of the trained neural network (default: ``./temp_edge``)
- ``nmin``: minimum graph size (default: 200)
- ``nmax``: maximum graph size (default: 5000)
- ``ntrain``: number of training graphs (default: 10000)
- ``epochs``: number of epochs (default: 1)
- ``print_rew``: steps to take before printing the reward (default: 1000)
- ``batch``: steps to take before updating the loss function (default: 8)
- ``hops``: number of hops (default: 3)
- ``workers``: number of workers (default: 8)
- ``lr``: learning rate (default: 0.001)
- ``gamma``: discount factor (default: 0.9)
- ``coeff``: critic loss coefficient (default: 0.1)
- ``units``: number of units in the graph convolutional layers (default: 7)
- ``dataset``: dataset type to choose between ``'delaunay'`` and ``'suitesparse'`` (default: ``'delaunay'``). With the first choice, random Delaunay graphs in the unit square are generated before the training. With the second choice, the user needs to download the matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) in the Matrix Market format and put the ``.mtx`` files in the folder ``drl-graph-partitioning/suitesparse_train``. In the paper we focused on matrices coming from 2D/3D discretizations.

After the training the weights are saved in ``out`` with the name ``model_separator_delaunay``. 

### Testing the DRL agent on several datasets
Run ``drl_separator_test.py`` with the following arguments
- ``out``: output folder to save the weights of the trained neural network (default: ``./temp_edge``)
- ``nmin``: minimum graph size (default: 100)
- ``nmax``: maximum graph size (default: 10000)
- ``ntest``: number of testing graphs (default: 1000)
- ``hops``: : number of hops (default:3)
- ``units``: number of units in the graph convolutional layers in the loaded refining DNN (default: 7)
- ``attempts``:  number of attempts to make (default: 3)
- ``dataset``: dataset type to choose among ``'delaunay'``, ``'suitesparse'``, and the Finite Elements triangulations ``graded_l``, ``hole3``, ``hole6`` (default: ``'delaunay'``). With the first choice, random Delaunay graphs in the unit square are generated before the evaluation. With the second choice, the user needs to download the matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) in the Matrix Market format and put the ``.mtx`` files in the folder ``drl-graph-partitioning/suitesparse``. For the Finite Elements triangulations, the user can download the matrices from [here](https://portal.nersc.gov/project/sparse/strumpack/fe_triangulations.tar.xz) and put the 3 folders in ``drl-graph-partitioning/``.

For each graph in the dataset the normalized separator computed with DRL and METIS is returned.

### DRL agent for the nested dissection ordering
Run ``drl_nd_testing.py`` with the following arguments
- ``out``: output folder to save the weights of the trained neural network (default: ``./temp_edge``)
- ``nmin``: minimum graph size (default: 100)
- ``nmax``: maximum graph size (default: 10000)
- ``ntest``: number of testing graphs (default: 1000)
- ``hops``: : number of hops (default: 3)
- ``units``: number of units in the graph convolutional layers in the loaded refining DNN (default: 7)
- ``attempts``: number of attempts to make (default: 3)
- ``dataset``: dataset type to choose among ``'delaunay'``, ``'suitesparse'``, and the Finite Elements triangulations ``graded_l``, ``hole3``, ``hole6`` (default: ``'delaunay'``). With the first choice, random Delaunay graphs in the unit square are generated before the evaluation. With the second choice, the user needs to download the matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) in the Matrix Market format and put the ``.mtx`` files in the folder ``drl-graph-partitioning/suitesparse``. For the Finite Elements triangulations, the user can download the matrices from [here](https://portal.nersc.gov/project/sparse/strumpack/fe_triangulations.tar.xz) and put the 3 folders in ``drl-graph-partitioning/``.

For each graph in the dataset it is returned the number of non-zero entries in the LU factorization of the associated adjacency matrix computed with DRL, METIS_ND, COLAMD, METIS, SCOTCH.

## Required software
- Pytorch
- Pytorch Geometric
- NetworkX
- Numpy
- Scipy
- NetworkX-METIS
