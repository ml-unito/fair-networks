# fair-networks

Let us consider a matrix $\matrix{X} \in \mathbb{R}^{n \times m}$, and two vectors $\vec{y} \in \mathbb{R}^n$ and $\vec{g} \in \mathbb{R}^n$. We consider the problem of building a model $M$ such that $M(\matrix{X}) \simeq \vec{y}$, with the constraint that model $M$ does not use any information useful to predict $\vec{g}$.

Let us then consider the network architecture shown below:

<img src="fair-network-img.png">

The idea is to:

1. train N,y-predictor, g-predictor so to optimize the predictions about $y$ and $g$. In this phase the network strives to predict $g$ as well as possible.
2. we want now to tweak things so that g-predictor cannot predict $g$ any more. To do so, we tweak N weights so to maintain high accuracy on $y$ while making g-predictor to be indistinguishable from a random guess.
3. now we want to give a chance to $g-predictor$ to exploit the remaining information in $N$ to predict $g$ so we re-train only y-predictor and g-predictor and in both cases we try to maximizes the performances.
4. we repeat 2 and 3 until step-3 is unable to find a good predictor for $g$.

Upon convergence we conclude that:

1. there is enough information on the last layer of $N$ to predict $g$;
2. if $y$ is still good enough, we obtained a classifier whose prediction does not depend on $g$ nor any information about $g$ that can be reconstructed from $X$.
