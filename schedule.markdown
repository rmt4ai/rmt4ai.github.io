---
layout: page
title: Workshop Schedule
permalink: /schedule/
---

## Talks

### Wedenesday, June 5, 2024

#### **8:00 – 8:45** Registration
#### **8:45 – 9:00** Welcoming remarks
#### **9:00 – 10:00** *(Keynote)* Marina Meila 
#### **10:00 – 10:20** Break
#### **10:20 – 11:05** Zhou Fan
#### **11:05 - 11:50** Liza Rebrova
#### **11:50 – 1:00** Lunch
#### **1:00 – 1:45** Alex Cloninger

**Deep learning based two sample tests with small data and small networks**

#### **1:45 - 2:30** Adit Radha
#### **2:30 – 2:50** Break
#### **2:50 – 3:35** Zhichao Wang 
#### **3:35 - 4:20** David Gamarnik
#### **4:20 – 4:40** Break and poster set up
#### **4:40 – 6:40** Reception and Poster Session

### Thursday, June 6, 2024

#### **8:00 – 9:00** Breakfast
#### **9:00 – 9:45** Nhat Ho

**Neural Collapse in Deep Neural Networks: From Balanced to Imbalanced Data**

*Abstract:* Modern deep neural networks have achieved impressive performance on tasks from image classification to natural language processing. Surprisingly, these complex systems with massive amounts of parameters exhibit the same structural properties in their last-layer features and classifiers across canonical datasets when training until convergence. In particular, it has been observed that the last-layer features collapse to their class means, and those class-means are the vertices of a simplex Equiangular Tight Frame (ETF). This phenomenon is known as Neural Collapse (NC). Recent papers have theoretically shown that NC emerges in the global minimizers of training problems with the simplified “unconstrained feature model”. In this context, we take a step further and prove the NC occurrences in deep neural networks for the popular mean squared error (MSE) and cross-entropy (CE) losses, showing that global solutions exhibit NC properties across the linear layers. Furthermore, we extend our study to imbalanced data for MSE loss and present the first geometric analysis of NC under a bias-free setting. Our results demonstrate the convergence of the last-layer features and classifiers to a geometry consisting of orthogonal vectors, whose lengths depend on the amount of data in their corresponding classes. 

#### **9:45 - 10:30** Bruno Loureiro
#### **10:30 – 10:45** Break
#### **10:45 – 11:30** Arthur Jacot

**Beyond the Lazy/Active Dichotomy: the Importance of Mixed Dynamics in Linear Networks**

#### **11:30 - 12:15** Katya Scheinberg
#### **12:15 – 1:30** Lunch
#### **1:30 – 2:50** Open problem session
#### **2:50 – 3:10** Break
#### **3:10 – 3:55** Courtney Paquette

**Scaling Law: Compute Optimal Curves on a Simple Model**

*Abstract:* We  describe a program of analysis of stochastic gradient methods on high dimensional random objectives.  We illustrate some assumptions under which the loss curves are universal, in that they can completely be described in terms of some underlying covariances.   Furthermore, we give description of these loss curves that can be analyzed precisely.   We show how this can be applied to SGD on a simple power-law model.  This is a simple two-hyperparameter family of optimization problems, which displays 4 distinct phases of loss curves; these phases are determined by the relative complexities of the target, data distribution, and whether these are ‘high-dimensional’ or not (which in context can be precisely defined).  In each phase, we can also give, for a given compute budget, the optimal parameter dimensionality. Joint work with Elliot Paquette (McGill), Jeffrey Pennington (Google Deepmind), and Lechao Xiao  (Google Deepmind).

#### **3:55 - 4:40** Denny Wu
#### **4:40 – 5:30** Small group discussions

### Friday, June 7, 2024

#### **8:00 – 9:00** Breakfast
#### **9:00 – 10:00** *(Keynote)* Michael Mahoney
#### **10:00 – 10:20** Break
#### **10:20 – 11:05** TBD
#### **11:05 - 11:50** [Mert Gürbüzbalaban](https://mert-g.org/) (Rutgers)

**Heavy Tail Phenomenon in Stochastic Gradient Descent**

*Abstract*: Stochastic gradient descent (SGD) methods are workhorse methods for training machine learning models, particularly in deep learning. After presenting numerical evidence demonstrating that SGD iterates with constant step size can exhibit heavy-tailed behavior even when the data is light-tailed, in the first part of the talk, we delve into the theoretical origins of heavy tails in SGD iterations based on analyzing products of random matrices and their connection to various capacity and complexity notions proposed for characterizing SGD's generalization properties in deep learning. Key notions correlating with performance on unseen data include the 'flatness' of the local minimum found by SGD (related to the Hessian eigenvalues), the ratio of step size η to batch size b (controlling stochastic gradient noise magnitude), and the 'tail-index' (which measures the heaviness of the tails of the eigenspectra of the network weights). We argue that these seemingly disparate perspectives on generalization are deeply intertwined. Depending on the Hessian structure at the minimum and algorithm parameter choices, SGD iterates converge to a heavy-tailed stationary distribution. We rigorously prove this claim in linear regression, demonstrating heavy tails and infinite variance in iterates even in simple quadratic optimization with Gaussian data. We further analyze tail behavior with respect to algorithm parameters, dimension, and curvature, providing insights into SGD behavior in deep learning. Experimental validation on synthetic data and neural networks supports our theory. Additionally, we discuss generalizations to decentralized stochastic gradient algorithms and to other popular step size schedules including the cyclic step sizes. In the second part of the talk, we introduce a new class of initialization schemes for fully-connected neural networks that enhance SGD training performance by inducing a specific heavy-tailed behavior in stochastic gradients. Based on joint work with Yuanhan Hu, Umut Simsekli, and Lingjiong Zhu. 

#### **11:50 – 1:00** Lunch

#### **1:00 – 2:30** Ending remarks and small group discussion


## Posters

*Information coming soon*
