# fair-networks

**Note**: use

```bash
 pandoc README.md -o README.pdf
```

to generate a PDF rendering this file (or open it in Atom and preview the markdown using the plugin `markdown-preview-plus`).

## A fair-network model

Let us consider a matrix $X \in \mathbb{R}^{n \times m}$, and two vectors: a target output vector $\vec{y} \in \mathbb{R}^n$ and a vector representing a sensible variable $\vec{s} \in \mathbb{R}^k$. We consider the problem of building a model $M$ such that $M(X) \simeq \vec{y}$, with the constraint that model $M$ does not use any information useful to predict $\vec{s}$.

Let us then consider the network architecture shown below:

![fair network image](images/fair-network-img.png "Fair network image")

Let $\mathcal{N}(X)$ bet the output of the network on input $X$. Let also be $\mathcal{N}_{\theta_0}$ be the network restricted to the portion governed by the $\theta_0$ parameters and define analogously $\mathcal{N}_{\theta_y}$ and $\mathcal{N}_{\theta_s}$.

The idea is as it follows:

- consider a new batch $X'$ of examples;
- the batch is used to compute a new representation $\chi$ for the examples as $\chi = N_{\theta_0}(X')$;
- train $\mathcal{N}_{\theta_y}$ and $\mathcal{N}_{\theta_y}$ over $\chi$ up to convergence using:
  - $$\arg\min_{\theta_y} L_{\theta_y}(y, \mathcal{N_{\theta_y}(\chi)})$$
  - $$\arg\min_{\theta_s} L_{\theta_s}(s, \mathcal{N_{\theta_s}(\chi)})$$
- evaluate $\sigma = \mathcal{N}_{\theta_s}(\chi)$
- evaluate $\gamma = \mathcal{N}_{\theta_y}(\chi)$
- update $\theta_0$ weights by back-propagating the error on the loss
  - $$L(y,\gamma, s, \sigma) = L_{\theta_y}(y, \gamma) - \lambda L_{\theta_s}(s,\sigma)$$
- where $\lambda$ is a parameter specifying how important is the fairness in the decision.

Upon convergence one evaluates the accuracy of the network in predicting $y$ and $s$:

-  if the accuracy on $s$ is low enough, one can conclude that there is not enough information on the last layer of $\theta_0$ to predict $s$;
- if the accuracy on $y$ is still good enough, we obtained a good enough classifier whose prediction does not depend on $s$ nor any information about $s$ that can be reconstructed from $X$.
