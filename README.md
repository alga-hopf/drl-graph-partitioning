# Graph Partitioning and Sparse Matrix Ordering usingReinforcement Learning and Graph Neural Networks

Deep reinforcement learning models to find a minimum normalized cut partitioning and a minimal vertex separator from https://arxiv.org/abs/2104.03546.

## Minimum normalized cut

Given a graph $G=(V,E)$, the goal is to find a partition $V=V_A \cup V_B$, $V_A\cap V_B=\emptyset$ such that the quantity $\text{cut}(G)\left( \frac{1}{\text{vol(A)}}+\frac{1}{\text{vol(A)}}\right)$ is minimized, where $\text{cut}(G)=\sum_{(v,w)\in E \\ v\in V_A,w\in V_B}1$ and $\text{vol}(X)=\sum_{v\in X}\deg(v)$. In order to do that, we train a Distributed Advantage Actor-Critic (DA2C) agent to refine the partitions obtained by interpolating back the partition on a coarser representation of the graph. More precisely, the initial graph is coarsened until it has a small number of nodes, then the coarsest graph is partitioned and the partition is interpolated back one level, where it is further refined. This process is applied recursively up to the initial graph. To partition the coarsest graph one can use the software METIS or train another deep reinforcement learning agent to partition it. The explained multilevel algorithm is summarized in the picture below. Both agents are implemented by two-headed deep neural network containing graph convolutional layers.

![Multilevel graph partitioning](/images/coarsening_refining.png)

The codes for the training (refining and partitioning the coarsest graph) and for the evaluation are in the partitioning folder. 

## Minimal vertex separator

## Required packages
- Pytorch
- Pytorch Geometric
- NetworkX
- Numpy
- Scipy
- NetworkX-METIS
