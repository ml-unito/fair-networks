# fair-networks

**Note**: use

```bash
 pandoc README.md -o README.pdf
```

to generate a PDF rendering this file (or open it in Atom and preview the markdown using the plugin `markdown-preview-plus`).

## A fair-network model

Let us consider a matrix $X \in \mathbb{R}^{n \times m}$, and two vectors $\vec{y} \in \mathbb{R}^n$ and $\vec{g} \in \mathbb{R}^n$. We consider the problem of building a model $M$ such that $M(X) \simeq \vec{y}$, with the constraint that model $M$ does not use any information useful to predict $\vec{g}$.

Let us then consider the network architecture shown below:

![fair network image](images/fair-network-img.png "Fair network image")

The idea is to:

- train N,y-predictor, g-predictor so to optimize the predictions about $y$ and $g$. In this phase the network strives to predict $g$ as well as possible.
- we want now to tweak things so that g-predictor cannot predict $g$ any more. To do so, we tweak N weights so to maintain high accuracy on $y$ while making g-predictor to be indistinguishable from a random guess.
- now we want to give a chance to $g-predictor$ to exploit the remaining information in $N$ to predict $g$ so we re-train only y-predictor and g-predictor and in both cases we try to maximizes the performances.
- we repeat 2 and 3 until step-3 is unable to find a good predictor for $g$.

Upon convergence we conclude that:

- there is enough information on the last layer of $N$ to predict $g$;
- if $y$ is still good enough, we obtained a classifier whose prediction does not depend on $g$ nor any information about $g$ that can be reconstructed from $X$.

Cheers
