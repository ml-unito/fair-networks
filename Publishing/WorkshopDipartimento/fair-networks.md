

# Fair-Networks
## Constraining neural networks representations to enforce fair decisions

---

# Problem statement

Build learning algorithms that provide some guarantee that no sensible information is used

---

# Motivations

Machine Learning is experiencing e spike in interest umparalleled in his history.
Machine Learning techniques are being deployed by big companies to solve all sort of problems: [list of problems]

When the decisions to be made impact the quality of life of people in the millions, it is imperative to guarantee the fairness of the decision making process.

---

# Approaches not based on neural networks

---

# Why neural networks?

- it is a buzzword in the business world (for good reasons too), many companies will attempt deploying neural networks regardless the existence of fair options;
- neural networks have the pecularity of being modular objects: they are naturally fit for building representations that can be then exploited by other neural networks or by other models;
- the problem of verifying if the output of a "fair classifier" is indeed fair is a hard problem per se. This property is easier to be empirically verified in NN (for reasons that we will see).

---

# Empirical Fairness Tests

- Other approaches compare the output on the target variable $y$ with some "fair" output gathered somehow. Unfortunately creating such labelings is extremely difficult and poses problems that go far beyound the usual problem of labeling a dataset: how do you guarantee that the labelers are not themselves biased w.r.t. the sensible attribute?
- In our approach we exploit the modularity of neural networks by forcing the network to build a "fair" representation. We are then in the position to test the fairness of such representation by trying to predict the sensible variable. If we can't succeed regardless of how hard we try and of the technique used: we are in a strong position to argue that the prediction of $y$ is itself independent on $s$.


---


# Other approaches

VAEs



<!--

ABSTRACT: The recent surge in interest for Deep Learning (motivated by its exceptional performances on long standing problems) made Neural Networks a very appealing tool for many actors in our society. Companies that traditionally used more explainable tools are nowadays considering Neural Networks to attack the problems they face everyday. One problem in this shift of interest is that Neural Networks are very opaque objects and more often than not the reasons underlying the results they provide are unclear to the final decision maker. When the decisions to be made impact the quality of life of people in the millions, it is imperative to guarantee the fairness of the decision making process: i.e., that the decision is made without discriminating people on a number of sensible attributes such as gender, sex orientation, religion, etc.

Fair-Networks are a tentative to mitigate the problem of using Neural Networks in such context: by explicitly modelling a preference for not using the sensible attributes, we build networks that construct representations of the data where any information about sensible attributes (either direct or mediated) is eliminated. By constructing classifiers on top of that representations one can guarantees that better, fairer, decisions will be made.

-->