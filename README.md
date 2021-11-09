# Wasserstein-Distance

### Tutorials
1. [Optimal Transport for Applied Mathematicians](http://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf)
2. [Optimal Transport for Data Analysis](https://www.uni-muenster.de/AMM/num/Vorlesungen/OptTransp_SS17/ss2017_OTDataAnalysis_2017-05-02.pdf)
3. [A user's guide to optimal transport](https://www.math.umd.edu/~yanir/OT/AmbrosioGigliDec2011.pdf)




### Computation

#### Discretization 
Minimize $\quad \sum_{i=1}^{m} \sum_{j=1}^{n} c_{i j} x_{i j}$
Subject to:
$$
\begin{array}{l}
\sum_{j=1}^{n} x_{i j}=a_{i} \quad \text { for } i=1,2, \ldots, m \\
\sum_{i=1}^{m} x_{i j}=b_{j} \quad \text { for } j=1,2, \ldots, n \\
x_{i j} \geq 0 \quad \text { for } i=1,2, \ldots, m \text { and } j=1,2, \ldots, n
\end{array}
$$
1. Linear programming
2. Sinkhorn iteration
   1. [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)
3. ADMM
   1. [Fast Discrete Distribution Clustering Using Wasserstein Barycenter With Sparse Support](https://arxiv.org/abs/1510.00012)
   2. [A Fast Globally Linearly Convergent Algorithm for the
Computation of Wasserstein Barycenters](https://www.jmlr.org/papers/volume22/19-629/19-629.pdf)



#### Lipschitz constraint
$$
\mathcal{W}[p, q]=\max _{f}\left\{\int[p(x) f(\boldsymbol{x})-q(\boldsymbol{x}) f(\boldsymbol{x})] d x \mid\|f\|_{L} \leq 1\right\}
$$
1. Wasserstein Generative Adversarial Networks
2. Improved Training of Wasserstein GANs

#### C-transform
1. Three-Player Wasserstein GAN via Amortised Duality

    ##### Quadratic + ICNN 
    1. 2-Wasserstein Approximation via Restricted Convex Potentials with Application to Improved Training for GANs
    2. Optimal transport mapping via input convex neural networks
    3. Input Convex Neural Networks
    4. Optimal Control Via Neural Networks: A Convex Approach

#### 2-step Computation
1. A Two-Step Computation of the Exact GAN Wasserstein Distance
2. Wasserstein GAN with Quadratic Transport Cost

#### GMM
1. [Optimal transport for Gaussian mixture models](https://arxiv.org/pdf/1710.07876.pdf)

#### Empirical study
1. Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark






