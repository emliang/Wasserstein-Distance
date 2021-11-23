# Wasserstein-Distance
[link](htmlpreview.github.io/https://github.com/emliang/Wasserstein-Distance)
This is an brief introduction to Wasserstein-Distance, including its formulation, computation and application.
<font size=2>

- [Wasserstein-Distance](#wasserstein-distance)
    - [Tutorials](#tutorials)
    - [Introduction](#introduction)
      - [Existing metrics](#existing-metrics)
      - [Transportation problem](#transportation-problem)
    - [Formulation](#formulation)
      - [Wasserstein distance (Kantorovich formulation)](#wasserstein-distance-kantorovich-formulation)
      - [Optimal transport and Wasserstein distance](#optimal-transport-and-wasserstein-distance)
      - [Several dual formulations](#several-dual-formulations)
        - [Kantorovich-Rubinstein Duality](#kantorovich-rubinstein-duality)
        - [Lipschitz constrained formulation](#lipschitz-constrained-formulation)
        - [Unconstrained formulation](#unconstrained-formulation)
        - [Quadratic cost function](#quadratic-cost-function)
        - [Convex formulation](#convex-formulation)
    - [Computation](#computation)
      - [Close form](#close-form)
      - [Discrete case](#discrete-case)
        - [Linear programming](#linear-programming)
        - [Sinkhorn iteration](#sinkhorn-iteration)
        - [ADMM](#admm)
      - [Continue case](#continue-case)
        - [Two-step computation](#two-step-computation)
        - [Penalty term](#penalty-term)
        - [Lipschitz constraint](#lipschitz-constraint)
        - [Unconstrained optimization](#unconstrained-optimization)
        - [Convex formulation](#convex-formulation-1)
        - [Gaussian mixture model](#gaussian-mixture-model)
      - [Empirical study](#empirical-study)
    - [Application](#application)



### Tutorials
1. [Optimal Transport for Applied Mathematicians](http://math.univ-lyon1.fr/~santambrogio/OTAM-cvgmt.pdf)
2. [Optimal Transport for Data Analysis](https://www.uni-muenster.de/AMM/num/Vorlesungen/OptTransp_SS17/ss2017_OTDataAnalysis_2017-05-02.pdf)
3. [A user's guide to optimal transport](https://www.math.umd.edu/~yanir/OT/AmbrosioGigliDec2011.pdf)


### Introduction
We will start from some some intuitive examples.
#### Existing metrics
1. metrics
   1. KL divergence: $D_{\mathrm{KL}}(P \| Q)=-\sum_{i} P(i) \ln \frac{Q(i)}{P(i)}$
   2. JS divergence: $D_{\mathrm{JS}}(P, Q)=\frac{1}{2}\left(D_{\mathrm{KL}}\left(P \| \frac{P+Q}{2}\right)+D_{\mathrm{KL}}\left(Q \| \frac{P+Q}{2}\right)\right)$
   3. F divergence: $D_{f}(p \| q)=\int q(x) f\left(\frac{p(x)}{q(x)}\right) d x$, where $f$ is a convex function.
2. Issues
   1. can not evaluate 2 distributions with different support set. 
      1. example: $\{p(x)|x\in[0,1]\}$ and $\{q(y)| y\in[2,3]\}$
   2. use KL/JS divergence as loss function -> gradient vanishing!
   3. need well-defined distance metric

#### Transportation problem
$$\begin{array}{|c||c|c|c|c||c|}
\hline & D_{1} & D_{2} & \cdots & D_{n} & \text { Supply } \\
\hline \hline O_{1} & c_{11} & c_{12} & \cdots & c_{1 n} & a_{1} \\
\hline O_{2} & c_{21} & c_{22} & \cdots & c_{2 n} & a_{2} \\
\hline \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
\hline O_{m} & c_{m 1} & c_{m 2} & \cdots & c_{m n} & a_{m} \\
\hline \hline \text { Demand } & b_{1} & b_{2} & \cdots & b_{n} & \\
\hline
\end{array}$$
1. Problem:
   1. origin: $\{O_1,...,O_m\}$, destination: $\{D_1,...,D_n\}$
   2. supply: $\{a_1,...,a_m\}$, demand: $\{b_1,...,b_n\}$
   3. transport supply goods in origin locations to satisfy demand in destinations
   4. transport plan/matrix: $x_{ij}$
   5. transport cost: $c_{ij}$
2. Formulation
   1. Primal formation
Minimize $\quad \sum_{i=1}^{m} \sum_{j=1}^{n} c_{i j} x_{i j}$
Subject to:
$$\begin{array}{l}
\sum_{j=1}^{n} x_{i j}=a_{i} \quad \text { for } i=1,2, \ldots, m \\
\sum_{i=1}^{m} x_{i j}=b_{j} \quad \text { for } j=1,2, \ldots, n \\
x_{i j} \geq 0 \quad \text { for } i=1,2, \ldots, m \text { and } j=1,2, \ldots, n
\end{array}$$
   1. Dual formulation
Maximize $\quad \sum_{i=1}^{m} a_{i} f_{i}+\sum_{i=1}^{n} b_{i} g_{i}$
Subject to:
$$f_{i}+g_{j} \leq c_{i j} \quad \text { for } i=1,2, \ldots, m, \text { for } j=1,2, \ldots, n$$


### Formulation
#### Wasserstein distance (Kantorovich formulation)
   1. view it as a continuous version of transportation problem
![](/pic/2021-11-22-19-52-57.png)
Minimize $\mathcal{W}[p, q]=\inf _{\gamma \in \Pi[p, q]} \iint \gamma( {x},  {y}) c( {x},  {y}) d  {x} d  {y}$
Subject to:
$$
\begin{array}{l}
\int \gamma( {x},  {y}) d  {y}=p( {x}) \\
\int \gamma( {x},  {y}) d  {x}=q( {y})
\end{array}
$$
   1. cost function $c(x,y)$: 
      1. any norm, $\| {x}- {y}\|_{1}, \;\| {x}- {y}\|_{2}, \;\| {x}- {y}\|_{2}^{2}$
   2. joint distribution $\gamma(x,y)$: 
      1. with marginal distribution $\gamma(x)=p(x),\gamma(y)=q(y)$
#### Optimal transport and Wasserstein distance
   1. Optimal transport (Monge formulation)
![](/pic/2021-11-22-19-56-38.png)
$$\begin{array}{c}
C_{M}(T)=\int_{\Omega} c(x, T(x)) p(x) \mathrm{d} \\
q=T(p)
\end{array}$$
      1. transport map: $q(y)=T(p(x))$
         1. non-linear constraint
#### Several dual formulations
##### Kantorovich-Rubinstein Duality
$$\mathcal{W}[p, q]=\max _{f, g}\left\{\int[p(x) f( {x})+q( {x}) g( {x})] d x \mid f( {x})+g( {y}) \leq c( {x},  {y})\right\}$$
      1. primal-dual optimality condition 
         1. $f( {x})+g( {y})=c( {x},  {y})$
         2. Proof: 
            1. forward: if primal and dual reach optimality, then
   $$\begin{align}
&\underbrace{\iint \gamma( {x},  {y}) c( {x},  {y}) d  {x} d  {y}}_{\text{primal formulation}}\\
=&\underbrace{\int[p(x) f( {x})+q( {x}) g( {x})] d x}_{\text{primal}=\text{dual} \text{ when reaching optimality}}\\
=&\underbrace{\iint[f(x)+g(y)] \gamma(x, y) d x d y}_{\text{marginal distribution}}\\
& \rightarrow f( {x})+g( {y})=c( {x},  {y}) \end{align}$$ 
            2. backward: if $f( {x})+g( {y})=c( {x},  {y})$ holds, then:
$$\begin{align}
& \underbrace{\int[p(x) f( {x})+q({x}) g( {x})] d x}_{\text{dual formulation}}\\
=&\underbrace{\iint[f(x)+g(y)] \gamma(x, y) d x d y}_{\text{marginal distribution}}\\
=&\underbrace{\iint \gamma( {x},  {y}) c( {x},  {y}) d  {x} d  {y}}_{f( {x})+g( {y})=c( {x},  {y})}\\
&\rightarrow \text{primal}=\text{dual} \text{ when reaching optimality}\end{align}$$
##### Lipschitz constrained formulation
$$\mathcal{W}[p, q]=\max _{f}\left\{\int[p(x) f(\boldsymbol{x})-q(\boldsymbol{x}) f(\boldsymbol{x})] d x \mid\|f\|_{L} \leq 1\right\}$$
   1. consider the optimality condition when $x=y$
      1. $f(y)+g(y)=c(y,y)=0\rightarrow g(y)=-f(y)$
   2. take $g(y)=-f(y)$ into the $\mathcal{W}[p, q]$
      1. objective function: $\max _{f}\left\{\int[p(x) f(\boldsymbol{x})-q(\boldsymbol{x}) f(\boldsymbol{x})] d x \right\}$
      2. constraints: $\|f\|_{L} \leq 1$
         1. $f(x)-f(y) \leq c(x, y)$ and $f(y)-f(x) \leq c(y, x)$
         2. $\|f\|_{L}=\frac{|f(x)-f(y)|}{c(x, y)} \leq 1$
##### Unconstrained formulation
$$\mathcal{W}[p, q]=\max _{f} \int f(x) d p(x)+\int \min _{x}[c(x, y)-f(x)] d q(y)$$
1. C-transform:
   1. For $f \in C(\Omega)$ define its $c$-transform $f^{c} \in C(\Omega)$ by
$$
f^{c}(y)=\inf \{c(x, y)-f(x) \mid x \in \Omega\}
$$
   2. and its $\bar{c}$-transform $g^{\bar{c}} \in C(\Omega)$ by
$$
g^{\bar{c}}(x)=\inf \{c(x, y)-g(y) \mid y \in \Omega\}
$$
   3. $f^{c\hat{c}}(x)\ge f(x)$, "$=$" holds when $f$ is concave
1. Consider $g(y)$ is the C-transform of $f(x)$
   1. $f^{c}(y)=\inf_x \{c(x, y)-f(x) \}$
   2. Proof of such a transform will not affect the optimality
      1. prove $f(x)$ and $f^c(y)$ satisfy the constraint
         $$\begin{align}
         & f(x)+\inf \{c(x, y)-f(x)\}\\
         \leq &  f(x)+c(x, y)-f(x)\\
         =   & c(x, y) 
         \end{align}$$
         The constraint is always be satisfied under C-transform
      2. prove $f(x)$ and $f^c(y)$ reach optimality condition
         $$\begin{align}
         & f(x)=g^{c}(x)\\
         \rightarrow &  f^{c}(y)=g^{c \hat{c}}(y) \geq g(y)\\
         \rightarrow & f(x)+f^{c}(y) \geq f(x)+g(y)\\
         \end{align}$$

         when $f(x)+g(y)=c(x,y)$, $c(x,y)\leq f(x)+f^{c}(y)\geq c(x,y)$
         Therefore $f(x)+f^{c}(y)=c(x,y)$ and reaches optimality.
##### Quadratic cost function
1. quadratic cost function: $c(x,y)=\frac{1}{2}\|x-y\|^2$
2. The C-transform can be simplified as:
$$
\begin{aligned}
f(x) &=\inf_y \left\{\frac{1}{2}\|x-y\|^{2}-g(y) \right\} \\
&=\frac{1}{2}\|x\|^{2}+\inf_y \left\{-\langle x, y\rangle+\frac{1}{2}\|y\|^{2}-g(y) \right\} \\
&=\frac{1}{2}\|x\|^{2}-\underbrace{\sup_y \left\{\langle x, y\rangle-\left[\frac{1}{2}\|y\|^{2}-g(y)\right] \right\}}_{:=\phi(x): \operatorname{convex}}
\end{aligned}
$$
   1. $\phi(x)$ is the convex conjugate of $\frac{1}{2}\|y\|^{2}-g(y)$
1. Brenier theorem: 
   1. Under quadratic case, optimal transport map $T(x)$ is equivalent with transport plan $\gamma(x,y)$
$$T(x)=x-\nabla f(x)=x-(x-\nabla \phi(x))=\nabla \phi(x)$$
##### Convex formulation
$$\begin{array}{l}
\mathcal{W}[p, q]=C_{p, q}-\min\limits_{f'\in \text{cvx}}\max\limits_{g'\in \text{cvx}} \left\{\mathbb{E}_{p}[f'(x)]+\mathbb{E}_{q}[f^{'*}(y)] \right\}\\
\end{array}$$
1. under quadratic case: 
$$\begin{array}{l}
f(x)+g(y) \leq \frac{1}{2}\|x-y\|_{2}^{2} \Longleftrightarrow \\
{\left[\frac{1}{2}\|x\|_{2}^{2}-f(x)\right]+\left[\frac{1}{2}\|y\|_{2}^{2}-g(y)\right] \geq \langle x, y\rangle}
\end{array}$$
1. define:
   1. $f'(x)=\frac{1}{2}\|x\|_{2}^{2}-f(x)$
   2. $g'(y)=\frac{1}{2}\|y\|_{2}^{2}-g(y)$
2. The objective function becomes:
$$\begin{array}{l}
\mathcal{W}[p, q]=C_{p, q}-\min_{f',g'} \left\{\mathbb{E}_{p}[f'(x)]+\mathbb{E}_{q}[g'(y)]\mid f'(x)+g'(y)\ge \langle x, y\rangle \right\}\\
C_{p, q}=\frac{1}{2} \mathbb{E}_p[\|X\|_{2}^{2}]+ \mathbb{E}_q[ \|Y\|_{2}^{2}]
\end{array}$$
4. apply the conjugate transformation
   1. $g'(y)=f^{'*}(y)=\sup_x \left\{\langle x, y\rangle-\underbrace{\left[\frac{1}{2}\|x\|^{2}-f(x)\right]}_{f'(x)} \right\}$ 
5. unconstrained optimization
$$\begin{array}{l}
\mathcal{W}[p, q]=C_{p, q}-\min_{f',g'} \left\{\mathbb{E}_{p}[f'(x)]+\mathbb{E}_{q}[f^{'*}(y)] \right\}\\
\end{array}$$
   1. similar proof as C-transform
      1. constraint:
         1. $f'(x)+f^{'*}(y)\ge \langle x, y\rangle$
      2. optimality
         1. $f^{**}\leq f$, "$=$" holds when $f$ is convex
         2. $f'$ and $g'$ are convex
6. According to the Brenier theorem, when reach optimality
   1. $x=\nabla g'(y)= T(y)$ is the optimal transport map
   2. $f^{'*}(y) = \sup_x \left\{\langle x, y\rangle-{\left[\frac{1}{2}\|x\|^{2}-f(x)\right]} \right\}$
   3. $f^{*'}(y)= \langle T(y), y\rangle-{\left[\frac{1}{2}\|T(y)\|^{2}-f'(T(y))\right]}$
7. convex formulation:
$$\begin{array}{l}
\mathcal{W}[p, q]=C_{p, q}-\min\limits_{f'\in \text{cvx}}\max\limits_{g'\in \text{cvx}} \left\{\mathbb{E}_{p}[f'(x)]+\mathbb{E}_{q}[f^{'*}(y)] \right\}\\
\mathcal{W}[p, q]=C_{p, q}-\min\limits_{f'\in \text{cvx}}\max\limits_{g'\in \text{cvx}} \left\{\mathbb{E}_{p}[f'(\nabla g'(y))]+\mathbb{E}_{q}[\langle \nabla g'(y), y\rangle-f'(\nabla g'(y))] \right\}\\
\end{array}$$


### Computation

#### Close form
1. Gaussian distribution under quadratic cost
   1. Distributions: $ \mathcal{N}_{1}\left(\mu_{1}, \Sigma_{1}\right) $, $ \mathcal{N}_{2}\left(\mu_{2}, \Sigma_{2}\right)$
   2. Transport map: $x \longrightarrow \mu_{2}+A\left(x-\mu_{1}\right)$
         1. $A=\Sigma_{1}^{-1 / 2}\left(\Sigma_{1}^{1 / 2} \Sigma_{2} \Sigma_{1}^{1 / 2}\right) \Sigma_{1}^{-1 / 2}$
   3. W-distance: 
$$
W_{2}\left(\mathcal{N}_{1}, \mathcal{N}_{2}\right)=\left\|\mu_{1}-\mu_{2}\right\|_{2}^{2}+\operatorname{Tr}\left(\Sigma_{1}+\Sigma_{2}-2\left(\Sigma_{1}^{1 / 2} \Sigma_{2} \Sigma_{1}^{1 / 2}\right)^{1 / 2}\right)
$$

#### Discrete case

##### Linear programming
Maximize $\quad \sum_{i=1}^{n} a_{i} f_{i}+\sum_{i=1}^{n} b_{i} g_{i}$
Subject to:
$$
f_{i}+g_{j} \leq c_{i j} \quad \text { for } i=1,2, \ldots, n, \text { for } j=1,2, \ldots, n
$$
1. solve in dual form
2. polynomial complexity
3. not scalable when n is large

##### Sinkhorn iteration
Minimize $\quad \sum_{i=1}^{n} \sum_{j=1}^{n} c_{i j} x_{i j}-\lambda^{-1} H(x)$
Subject to:
$$
\begin{array}{l}
\sum_{j=1}^{n} x_{i j}=p_{i} \quad \text { for } i=1,2, \ldots, n \\
\sum_{i=1}^{n} x_{i j}=q_{j} \quad \text { for } j=1,2, \ldots, n \\
x_{i j} \geq 0 \quad \text { for } i, j=1,2, \ldots, n
\end{array}
$$
1. entropy regularization: $H(x)=-x\log x$
   1. strongly convex
   2. link primal and dual solution
2. Optimality condition
   1. $\nabla_x L(x,f,g) = 0 = c_{ij}+\lambda^{-1}(1+\log x)-f_i-g_j$
   2. $x_{ij}=e^{\lambda f_i} e^{-c_{ij}\lambda-1}e^{\lambda g_j}=v_iK_{ij}u_j$
   3. constraints
$$\begin{array}{ll}
\sum_{j=1}^{n} x_{i j}=v_i\sum_{j=1}^n K_{ij}u_j =p_{i} & \text { for } i=1,2, \ldots, n \\
\sum_{i=1}^{n} x_{i j}=u_j\sum_{i=1}^n v_iK_{ij}=q_{j} & \text { for } j=1,2, \ldots, n
\end{array}$$
1. Matrix normalization/balancing
   1. find a matrix has row and column constraints
   2. double stochastic matrix
2. Sinkhorn-Knopp algorithm
$$
\begin{aligned}
v_{i}^{n+1} &=\frac{p_{i}}{\sum_{j} K_{i j} u_{j}^{n}} \\
u_{j}^{n+1} &=\frac{p_{i}}{\sum_{i} K_{i j} v_{i}^{n+1}}
\end{aligned}
$$
3. limited numerical accuracy when $\lambda$ is large
##### ADMM 

#### Continue case

##### Two-step computation
##### Penalty term
##### Lipschitz constraint
##### Unconstrained optimization
##### Convex formulation
##### Gaussian mixture model

#### Empirical study

 

 


### Application

</font>

<!-- 


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
\mathcal{W}[p, q]=\max _{f}\left\{\int[p(x) f( {x})-q( {x}) f( {x})] d x \mid\|f\|_{L} \leq 1\right\}
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
1. Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark -->






