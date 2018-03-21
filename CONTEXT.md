---
output:
  html_document: default
  pdf_document: default
  word_document: default
---
# fair-networks - FAIRNESS AWARE LEARNING


## Context Introduction
With recent Artificial Intelligence and Machine Learning developments the use of profiling and automated decision-making has been increasing across many sectors like finance, insurance, marketing, ...

The General Data Protection Regulation expressly addresses the processing activities on Profiling and Automated Decision on WP29 New Guidelines.

The second part of the guidelines analyses how apply to general profiling activities, the general principles of the GDPR, such as: transparency, fairness, data minimisation, accuracy, purpose compatibility, and storage.

In particular the process have to guarantee:

- Non discrimination. Discrimination might be defined as the unfair treatment of an individual because of his or her membership in a particular group, e.g. race, gender, etc. Machine learning can reify existing patterns of discrimination-if they are found in the training dataset, then by design an accurate classifier will reproduce them. In this way, biased decisions are presented as the outcome of an 'objective' algorithm.

- Right explanation: standard supervised machine learning algorithms for regression or classification are inherently based on discovering reliable associations / correlations to aid in accurate out-of-sample prediction, with no concern for causal reasoning or "explanation" beyond the statistical sense in which it is possible to measure the amount of variance explained by a predictor.

##Introduction
Algorithmic decision making process becoming increasingly automated and data-driven; often to support human supervision in decision making decisions, but sometimes also to replace them (e.g. in Big Data scenarios). There are growing concerns about potential loss of transapercy, accountability and fairness.
Discrimination and unfair treatment of people based on certain attributes (e.g. sensitive attributes such generd or race)  are  are to be avoid.
[Barocas and Selbst 2016]Fairness of a decision making process could be defined with two different notions: disaparate treatment and disparate impact.
A decision making suffers from disparate treatment if its decision are based on the subject's sensitivie attribute information and it has disparate impact if its outcomes disproportionately hurf (or benefit) people with certain sensitive attribute values.
It's desiderable to design decision making system free of disparate treatment.




