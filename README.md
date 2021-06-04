# Graph Partitioning and Sparse Matrix Ordering using Reinforcement Learning and Graph Neural Networks

Deep reinforcement learning models to find a minimum normalized cut partitioning and a minimal vertex separator from https://arxiv.org/abs/2104.03546.

## Minimum normalized cut

Given a graph G=(V,E), the goal is to find a partition of the set of vertices V such that its normalized cut is minimized. This is known to be an NP-complete problem, hence we look for approximate solutions to it. In order to do that, we train a Distributed Advantage Actor-Critic (DA2C) agent to refine the partitions obtained by interpolating back the partition on a coarser representation of the graph. More precisely, the initial graph is coarsened until it has a small number of nodes, then the coarsest graph is partitioned and the partition is interpolated back one level, where it is further refined. This process is applied recursively up to the initial graph. To partition the coarsest graph one can use the software METIS or train another deep reinforcement learning agent to partition it. Both agents are implemented by two-headed deep neural network containing graph convolutional layers.

The codes for the training (refining and partitioning the coarsest graph) and for the evaluation are in the partitioning folder. The details about the constructions and the trainings can be found in Section 2 and Section 3 of the paper.

## Minimal vertex separator
...

## Required software
- Pytorch
- Pytorch Geometric
- NetworkX
- Numpy
- Scipy
- NetworkX-METIS
