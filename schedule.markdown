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

**Kronecker-product random matrices and a matrix least-squares problem**

*Abstract:* forthcoming.

#### **11:05 - 11:50** Liza Rebrova
#### **11:50 – 1:00** Lunch
#### **1:00 – 1:45** Arthur Jacot

**Beyond the Lazy/Active Dichotomy: the Importance of Mixed Dynamics in Linear Networks**

#### **1:45 - 2:30** Adit Radha

**How do neural networks learn features from data?**

*Abstract:* Understanding how neural networks learn features, or relevant patterns in data, for prediction is necessary for their reliable use in technological and scientific applications. We propose a unifying mechanism that characterizes feature learning in neural network architectures. Namely, we show that features learned by neural networks are captured by a statistical operator known as the average gradient outer product (AGOP). Empirically, we show that the AGOP captures features across a broad class of network architectures including convolutional networks and large language models. Moreover, we use AGOP to enable feature learning in general machine learning models through an algorithm we call Recursive Feature Machine (RFM). We show that RFM automatically identifies sparse subsets of features relevant for prediction and explicitly connects feature learning in neural networks with classical sparse recovery and low rank matrix factorization algorithms. Overall, this line of work advances our fundamental understanding of how neural networks extract features from data, leading to the development of novel, interpretable, and effective models for use in scientific applications.

#### **2:30 – 2:50** Break
#### **2:50 – 3:35** Zhichao Wang 

**Signal propagation and feature learning in neural networks**

*Abstract:* In this talk, I will first present some recent work for the extreme eigenvalues of sample covariance matrices with spiked population covariance. Extending previous random matrix theory, we will characterize the spiked eigenvalues outside the bulk distribution and their corresponding eigenvectors for a nonlinear version of the spiked covariance model. Then, we will apply this new result to deep neural network models. Many recent works have studied the eigenvalue spectrum of the Conjugate Kernel (CK) defined by the nonlinear feature map of a feedforward neural network. However, existing results only establish weak convergence of the empirical eigenvalue distribution and fall short of providing precise quantitative characterizations of the ''spike'' eigenvalues and eigenvectors that often capture the low-dimensional signal structure of the learning problem. Using our general result for spiked sample covariance matrices, we will give a quantitative description of how spiked eigenstructure in the input data propagates through the hidden layers of a neural network with random weights. As a second application, we can study a simple regime of representation learning where the weight matrix develops a rank-one signal component over gradient descent training and characterize the alignment of the target function with the spike eigenvector of the CK on test data. This analysis will show how neural networks learn useful features at the early stage of training. This is a joint work with Denny Wu and Zhou Fan.

#### **3:35 - 4:20** David Gamarnik

**A curious case of the symmetric binary perceptron model: Algorithms and algorithmic barriers**

*Abstract:* Symmetric binary perceptron is a random model of a perceptron where a classifier is required to stay within a symmetric interval around zero, subject to randomly generated data. This model  exhibits an interesting and puzzling property: the existence of a polynomial time algorithm for finding a solution (classifier) coincides with the presence of an extreme form of  clustering. The latter means that  most of the satisfying solutions are singletons separated by large distances. For the majority of other random constraint satisfaction problems of this kind, this typically suggests algorithmic hardness, which evidently is not  the case for the symmetric perceptron model.
 
In order to resolve this conundrum, we conduct a different solution space geometry analysis. We establish that the model exhibits a phase transition called  multi-overlap-gap property (m-OGP), and we show that  the onset of this property asymptotically matches the performance of the best known algorithms, such as the algorithms constructed by Kim and Rouche, and Bansal and Spencer.  Next, we establish that m-OGP is a barrier to large classes of algorithms exhibiting either stability or online features (or both).  We show that Kim-Rouche and Bansal-Spencer algorithms indeed exhibit the stability and online features, respectively. We conjecture that m-OGP marks the onset of the genuine algorithmic hardness threshold for this model.
 
Joint work with Eren Kizildag (Columbia University), Will Perkins (Georgia Institute of Technology)  and Changji Xu (Harvard University).


#### **4:20 – 4:40** Break and poster set up
#### **4:40 – 6:40** Reception and Poster Session

### Thursday, June 6, 2024

#### **8:00 – 9:00** Breakfast
#### **9:00 – 9:45** Nhat Ho

**Neural Collapse in Deep Neural Networks: From Balanced to Imbalanced Data**

*Abstract:* Modern deep neural networks have achieved impressive performance on tasks from image classification to natural language processing. Surprisingly, these complex systems with massive amounts of parameters exhibit the same structural properties in their last-layer features and classifiers across canonical datasets when training until convergence. In particular, it has been observed that the last-layer features collapse to their class means, and those class-means are the vertices of a simplex Equiangular Tight Frame (ETF). This phenomenon is known as Neural Collapse (NC). Recent papers have theoretically shown that NC emerges in the global minimizers of training problems with the simplified “unconstrained feature model”. In this context, we take a step further and prove the NC occurrences in deep neural networks for the popular mean squared error (MSE) and cross-entropy (CE) losses, showing that global solutions exhibit NC properties across the linear layers. Furthermore, we extend our study to imbalanced data for MSE loss and present the first geometric analysis of NC under a bias-free setting. Our results demonstrate the convergence of the last-layer features and classifiers to a geometry consisting of orthogonal vectors, whose lengths depend on the amount of data in their corresponding classes. 

#### **9:45 - 10:30** Bruno Loureiro
#### **10:30 – 10:45** Break
#### **10:45 – 11:30** Alex Cloninger

**Deep learning based two sample tests with small data and small networks**

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
#### **10:20 – 11:05** Boris Hanin
#### **11:05 - 11:50** [Mert Gürbüzbalaban](https://mert-g.org/) (Rutgers)

**Heavy Tail Phenomenon in Stochastic Gradient Descent**

*Abstract*: Stochastic gradient descent (SGD) methods are workhorse methods for training machine learning models, particularly in deep learning. After presenting numerical evidence demonstrating that SGD iterates with constant step size can exhibit heavy-tailed behavior even when the data is light-tailed, in the first part of the talk, we delve into the theoretical origins of heavy tails in SGD iterations based on analyzing products of random matrices and their connection to various capacity and complexity notions proposed for characterizing SGD's generalization properties in deep learning. Key notions correlating with performance on unseen data include the 'flatness' of the local minimum found by SGD (related to the Hessian eigenvalues), the ratio of step size η to batch size b (controlling stochastic gradient noise magnitude), and the 'tail-index' (which measures the heaviness of the tails of the eigenspectra of the network weights). We argue that these seemingly disparate perspectives on generalization are deeply intertwined. Depending on the Hessian structure at the minimum and algorithm parameter choices, SGD iterates converge to a heavy-tailed stationary distribution. We rigorously prove this claim in linear regression, demonstrating heavy tails and infinite variance in iterates even in simple quadratic optimization with Gaussian data. We further analyze tail behavior with respect to algorithm parameters, dimension, and curvature, providing insights into SGD behavior in deep learning. Experimental validation on synthetic data and neural networks supports our theory. Additionally, we discuss generalizations to decentralized stochastic gradient algorithms and to other popular step size schedules including the cyclic step sizes. In the second part of the talk, we introduce a new class of initialization schemes for fully-connected neural networks that enhance SGD training performance by inducing a specific heavy-tailed behavior in stochastic gradients. Based on joint work with Yuanhan Hu, Umut Simsekli, and Lingjiong Zhu. 

#### **11:50 – 1:00** Lunch

#### **1:00 – 2:30** Ending remarks and small group discussion


## Posters

*Information coming soon*
