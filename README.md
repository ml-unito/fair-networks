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

## Install Instructions

Run the following commands:

```bash
git clone https://github.com/ml-unito/fair-networks
cd fair-networks
export PYTHONPATH="packages/:$PYTHONPATH"
export PATH="bin:$PATH" # put the preceeding two lines in your ~/.bashrc for ease of use
virtualenv venv --python=python3.6
source venv/bin/activate
pip3 install -r requirements.txt
pip install theano
mkdir experiments/first
```

Then create a file ```experiments/first/config.json``` containing the following:

```json
{
    "learning_rate": 0.01,
    "schedule": "m30:c10",
    "hidden_layers": "50",
    "class_layers": "50",
    "sensible_layers": "50",
    "fairness_importance": 0.5,
    "dataset": "default",
    "batch_size": 128,
    "noise_type": "sigmoid_sep",
    "dataset_base_path": "../../data",
    "model_dir": "models/",
    "eval_data": null,
    "eval_stats": false,
    "random_seed": 42,
    "output": null,
    "resume_ckpt": null,
    "checkpoint": "models/model.ckpt",
    "save_model_schedule": "3000:100"
}
```

Lastly, to perform the experiment and print results run:

```bash
fair_networks experiments/first/config.json
make
cat experiments/first/performances.tsv
```

Many parameters in ```config.json``` are self explanatory, but a non-exhaustive list follows:

```schedule```: how many epochs the network will be trained for. The expected syntax is ```m[num_epochs]:c[ignored value]```
```hidden_layers```: how many layers and how many neurons should be included in the hidden layers of the network. There is no need to set the input size here. The expected syntax is ```[number of neurons in first layer:in second layer:...]```
```dataset```: a string describing the dataset you would like to work on

## Adding a dataset

Dataset wrappers included with the code are provided in the package ```packages/fair/datasets/```. 
Writing your own wrapper code for a new dataset should be easy enough by looking at any one of the other wrappers. 
You will also need to add some argument handling code in ```packages/fair/utils/options.py```.
Preprocessing functionalities are provided in ```packages/fair/datasets/dataset_base.py``` and should work for your own .csv file. 
Note that you can either provide a download path for the data or put it in the ```data/``` folder.
