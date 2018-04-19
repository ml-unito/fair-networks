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

The idea is to:

- train parameters $\theta_0, \theta_y$ and $\theta_s$ so to optimize the predictions about $y$ and $s$. In this phase the network strives to predict $s$ as well as possible.
- we want now to tweak things so that s-predictor cannot predict $s$ any more. To do so, we tweak $\theta_0$ weights so to maintain high accuracy on $y$ while making the s-predictor to be indistinguishable from a random guess.
- now we want to give a chance to $s-predictor$ to exploit the remaining information in $\theta_0$ to predict $s$ so we re-train only y-predictor and s-predictor and in both cases we try to maximizes the performances.
- we repeat 2 and 3 until step-3 is unable to find a good predictor for $s$.

Upon convergence we conclude that:

- there is not enough information on the last layer of $\theta_0$ to predict $s$;
- if $y$ is still good enough, we obtained a classifier whose prediction does not depend on $s$ nor any information about $s$ that can be reconstructed from $X$.

More in the language of the neural networks. We envision an algorithm working like this:

- step 1: optimize everything for $y$ and try to predict $s$ given the result,
  - let $\theta^{(1)}_0, \theta^{(1)}_y = \arg\min_{\theta_0,\theta_y}{L(\theta_0,\theta_y)}$
  - let $\theta^{(1)}_s = \arg\min_{\theta_s} L(\theta_s | \theta_0)$

- step $n\in{2..T}$: optimize new parameters so to cripple the $s$ predictor
  - let $\theta^{(n)}_0 = \arg\min_{\theta_0} L^*(\theta_0 | \theta^{(n-1)}_s, \theta^{(n-1)}_y )$
    - where $L^*$ is devised to penalize good outputs from the $s$ predictor and incentivize good outputs from the $y$ predictor. It could be something like:
      $$
        L^*_s(\theta_0 | \theta_s) + \alpha L(\theta_0, \theta_y).
      $$
      In this description the hard part is defining $L^*_s$. One possibility is to define it as the distance to a random bit vector. If this implies changing $\theta_0$ to harshly we might try to change it in the smallest way possible while satisfying some minimum fairness contraint.
  - let $\theta^{(n)}_s = \arg\min_{\theta_s} L(\theta_s|\theta^{(n)}_0)$
  - let $\theta^{(n)}_y = \arg\min_{\theta_y} L(\theta_y|\theta^{(n)}_0)$
  - (note that the previous two points can be changed by combining learning $\theta_0$ with learning $\theta_y$ and only then optimize only for $\theta_s$ -- they appear almost the same thing, but in NN it is often the case that little differences produces vast changes in the outputs);


**Note**: invece della funzione che abbiamo deciso la volta scorsa e descritta qui sopra, si potrebbe usare l'approccio basato sulla covarianza proposto [nell'articolo che stiamo leggendo](https://pbpworkspace.slack.com/files/U7ZBR84N8/FA1N72RAN/16_fairness_constraints_w_annotations.pdf).

Cheers
