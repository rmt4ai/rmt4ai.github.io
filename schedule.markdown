---
layout: page
title: Workshop Schedule
permalink: /schedule/
---

## Talks

### Wednesday, June 5, 2024

####	8:00 - 8:45	Registration			
####	8:45 - 9:00	Welcoming Remarks	
####	9:00 - 10:00	Marina Meila (Keynote) – Manifold Coordinates with Physical Meaning
 
*Abstract:* We ask if it is possible, in the case of scientific data where quantitative prior knowledge is abundant, to explain a data manifold by new coordinates, chosen from a set of scientifically meaningful functions? The algorithm I will present, ManifoldLasso, can discover a subset of relevant coordinates from a user defined dictionary in fully non-parametric fashion. This is suppoerted by experiments on real data and theoretical recovery conditions.
 
Second, we ask how popular Manifold Learning tools and their applications can be recreated in the space of  vector fields and flows on a manifold. Central to this approach is the order 1-Laplacian, $\Delta_1$, whose eigen-decomposition into gradient, harmonic, and curl  provides a basis for all vector fields on a manifold. We present an estimator for $\Delta_1$, and based on it a new algorithm for finding shortest independent loops.
 
Joint work with  Yu-Chia Chen, Samson Koelle, Hanyu Zhang, Weicheng Wu and Ioannis Kevrekidis.
 

####	10:00 - 10:20	Break		
####	10:20 - 11:05	Liza Rebrova	 – On step size choices in stochastic and mini-batch gradient descent*

*Abstract:* First, I will talk about linear regression (and a little about ReLU regression). I will discuss robust stochastic gradient under the adversarial corruptions scenario and explain why exponentially decaying step size can be the right choice to ensure convergence. Then, for the least squares regression, I will discuss the connection between decreasing the mini-batch size when sampling without replacement, and decreasing the step size. These two changes have very similar effect on the convergence dynamic, but with subtle distinguishing effects that we propose to study via careful analysis of a certain anticommutator between sample covariance submatrices of the features. Based on the joint work with H. Jeong, D. Needell, J. Lok, and R. Sonthalia.

####	11:05 - 11:50 Zhou Fan – Kronecker-product random matrices and a matrix least-squares problem

We study the eigenvalue distribution and resolvent of a Kronecker-product random matrix model $A \otimes I_{n \times n}+I_{n \times n} \otimes B+\Theta \otimes \Xi \in \mathbb{C}^{n^2 \times n^2}$, where $A,B$ are independent Wigner matrices and $\Theta,\Xi$ are deterministic and diagonal. For fixed spectral arguments, we establish a quantitative approximation for the Stieltjes transform by that of an approximating free operator, and a diagonal deterministic equivalent approximation for the resolvent. We further obtain sharp estimates in operator norm for the $n \times n$ resolvent blocks, and show that off-diagonal resolvent entries fall on two differing scales of $n^{-1/2}$ and $n^{-1}$ depending on their locations in the Kronecker structure.

Our study is motivated by consideration of a matrix-valued least-squares optimization problem $\min_{X \in \mathbb{R}^{n \times n}} \frac{1}{2}\|XA+BX\|_F^2+\frac{1}{2}\sum_{ij} \xi_i\theta_j x_{ij}^2$ subject to a linear constraint. For random instances of this problem defined by Wigner inputs $A,B$, our analyses imply an asymptotic characterization of the minimizer $X$ and its associated minimum objective value as $n \to \infty$.

This is joint work with Jack (Renyuan) Ma.

####	11:50 - 1:00	Lunch Break			
####	1:00 -1:45 Arthur Jacot – Beyond the Lazy/Active Dichotomy: the Importance of Mixed Dynamics in Linear Networks	


####	1:45 - 2:30	Zhichao Wang	 – Signal propagation and feature learning in neural networks	

*Abstract:* In this talk, I will first present some recent work for the extreme eigenvalues of sample covariance matrices with spiked population covariance. Extending previous random matrix theory, we will characterize the spiked eigenvalues outside the bulk distribution and their corresponding eigenvectors for a nonlinear version of the spiked covariance model. Then, we will apply this new result to deep neural network models. Many recent works have studied the eigenvalue spectrum of the Conjugate Kernel (CK) defined by the nonlinear feature map of a feedforward neural network. However, existing results only establish weak convergence of the empirical eigenvalue distribution and fall short of providing precise quantitative characterizations of the ''spike'' eigenvalues and eigenvectors that often capture the low-dimensional signal structure of the learning problem. Using our general result for spiked sample covariance matrices, we will give a quantitative description of how spiked eigenstructure in the input data propagates through the hidden layers of a neural network with random weights. As a second application, we can study a simple regime of representation learning where the weight matrix develops a rank-one signal component over gradient descent training and characterize the alignment of the target function with the spike eigenvector of the CK on test data. This analysis will show how neural networks learn useful features at the early stage of training. This is a joint work with Denny Wu and Zhou Fan.

####	2:30 - 3:00	Break/Poster Setup	
####	3:00 - 3:45	Patrick Shafto (Special Session) – Artificial Intelligence Quantified (AIQ)

*Abstract:* The Artificial Intelligence Quantified (AIQ) Program will develop technology to assess and understand the capabilities of Artificial Intelligence (AI) to enable guaranteed performance. The program will test the hypothesis that mathematical methods, combined with advances in measurement and modeling, will allow guaranteed quantification of generative AI capabilities. Specifically, the program will address three capability levels: 1) specific problem level, which considers mapping between individual inputs and outputs, 2) classes of problem level, which considers collections of inputs and associated outputs, and 3) natural class level, which considers which inputs are well-behaved with respect to the outputs via choice of architecture and/or data, aiming to address the quantification and assessment challenges at each level. I will outline the AIQ program’s technical goals and challenges.

####	3:45 - 4:30	Open Problems/Discussion
####	4:30 - 6:30	Reception/Poster Session			
### Thursday, June 6, 2024

					
####	8:00 - 9:00	Breakfast			
####	9:00 - 10:00 Michael Mahoney (Keynote) –(Special Session) Practice, Theory, and Theorems for Random Matrix Theory in Modern Machine Learning

*Abstract:* Random Matrix Theory (RMT) has been applied to a wide range of areas over the years, and in recent years machine learning (ML) has been added to this list.  In many cases, this leads to new types of theory, either predictive theory or mathematical theorems.  Many aspects of modern ML are quite different than more traditional applications of RMT, and this is leading to new uses of and perspectives on RMT.  Here, we'll describe this, including both aspects of ML problem problem parameterization as well as empirical results on matrices arising in state-of-the-art ML models.  Based on this, we'll describe an RMT-based phenomenological theory that can be used, e.g., to predict trends in the quality of state-of-the-art neural networks without access to training or testing data.  This is starting to lead to new RMT theorems of independent interest, some of which we will also describe.

####	10:00 - 10:45 Break	
####	10:45 - 11:30 Adit Radha – How do neural networks learn features from data?**

*Abstract:* Understanding how neural networks learn features, or relevant patterns in data, for prediction is necessary for their reliable use in technological and scientific applications. We propose a unifying mechanism that characterizes feature learning in neural network architectures. Namely, we show that features learned by neural networks are captured by a statistical operator known as the average gradient outer product (AGOP). Empirically, we show that the AGOP captures features across a broad class of network architectures including convolutional networks and large language models. Moreover, we use AGOP to enable feature learning in general machine learning models through an algorithm we call Recursive Feature Machine (RFM). We show that RFM automatically identifies sparse subsets of features relevant for prediction and explicitly connects feature learning in neural networks with classical sparse recovery and low rank matrix factorization algorithms. Overall, this line of work advances our fundamental understanding of how neural networks extract features from data, leading to the development of novel, interpretable, and effective models for use in scientific applications.

####	11:30 - 12:15 Alex Cloninger – Deep learning based two sample tests with small data and small networks
####	12:15 - 1:30	Lunch			
####	1:30 - 2:15	David Gamarnik	– A curious case of the symmetric binary perceptron model: Algorithms and algorithmic barriers
	
*Abstract:* Symmetric binary perceptron is a random model of a perceptron where a classifier is required to stay within a symmetric interval around zero, subject to randomly generated data. This model  exhibits an interesting and puzzling property: the existence of a polynomial time algorithm for finding a solution (classifier) coincides with the presence of an extreme form of  clustering. The latter means that  most of the satisfying solutions are singletons separated by large distances. For the majority of other random constraint satisfaction problems of this kind, this typically suggests algorithmic hardness, which evidently is not  the case for the symmetric perceptron model.
 
In order to resolve this conundrum, we conduct a different solution space geometry analysis. We establish that the model exhibits a phase transition called  multi-overlap-gap property (m-OGP), and we show that  the onset of this property asymptotically matches the performance of the best known algorithms, such as the algorithms constructed by Kim and Rouche, and Bansal and Spencer.  Next, we establish that m-OGP is a barrier to large classes of algorithms exhibiting either stability or online features (or both).  We show that Kim-Rouche and Bansal-Spencer algorithms indeed exhibit the stability and online features, respectively. We conjecture that m-OGP marks the onset of the genuine algorithmic hardness threshold for this model.
 
Joint work with Eren Kizildag (Columbia University), Will Perkins (Georgia Institute of Technology)  and Changji Xu (Harvard University).


#### 2:15 - 3:00 Denny Wu – Two variants of learning single-index models with SGD

*Abstract:* Single-index models are given by a univariate link function applied to a one-dimensional projection of the input. Recent works have shown that the statistical complexity of learning this function class with online SGD is governed by the information exponent of the link function. In this talk, we discuss two variations of prior analyses. First, we consider the learning of single-index polynomials via SGD, but with reused training data. We show that two-layer neural networks optimized by an SGD-based algorithm can learn this target with almost linear sample complexity, regardless of the information exponent; this complexity surpasses the CSQ lower bound and matches the information-theoretic limit up to polylogarithmic factors. Next, we introduce the class of additive models defined as the sum of M single-index models, with M diverging with dimensionality d. We study the sample complexity of SGD training and also provide SQ lower bounds for learning this function class; our analysis reveals the fundamental difference between the previously studied finite-M (multi-index) and our large-M setting.  Based on joint works with Jason D. Lee, Kazusato Oko, Yujin Song, and Taiji Suzuki. 

####	3:00 - 3:20	Break			

####	3:20 - 4:05 Bruno Loureiro – [Learning features with two-layer neural networks, one step at a time](/slides/LoueiroBruno_DIMACS_Slides2024)

Feature learning - or the capacity of neural networks to adapt to the data during training - is often quoted as one of the fundamental reasons behind their unreasonable effectiveness. Yet, making mathematical sense of this seemingly clear intuition is still a largely open question. In this talk, I will discuss a simple setting where we can precisely characterise how features are learned by a two-layer neural network during the very first few steps of training, and how these features are essential for the network to efficiently generalise under limited availability of data.

Based on the following works: [1](https://arxiv.org/abs/2305.18270) and [2](https://arxiv.org/abs/2402.04980).

####	4:05 - 4:50 Nhat Ho – [Instability, Computational Efficiency and Statistical Accuracy](/slides/HoNhat_DIMACS_Slides2024)

*Abstract:* Many statistical estimators are defined as the fixed point of a data-dependent operator, with estimators based on minimizing a cost function being an important special case. The limiting performance of such estimators depends on the properties of the population-level operator in the idealized limit of infinitely many samples. We develop a general framework that yields bounds on statistical accuracy based on the interplay between the deterministic convergence rate of the algorithm at the population level, and its degree of (in)stability when applied to an empirical object based on n samples. Using this framework, we analyze both stable forms of gradient descent and some higher-order and unstable algorithms, including Newton’s method and its cubic-regularized variant, as well as the EM algorithm. We provide applications of our general results to several concrete classes of singular statistical models, including Gaussian mixture estimation, single-index models, and informative non-response models. We exhibit cases in which an unstable algorithm can achieve the same statistical accuracy as a stable algorithm in exponentially fewer steps—namely, with the number of iterations being reduced from polynomial to logarithmic in sample size n. 

This is based on joint work with Raaz Dwivedi, Koulik Khamaru, Tongzheng Ren, Purnamrita Sarkar, Sujay Sanghavi, Rachel Ward,  Martin J. Wainwright,  Michael I. Jordan,  Bin Yu.

			
### Friday, June 7, 2024

		
####	8:00 - 9:00	Breakfast			


####	9:00 - 9:45 Courtney Paquette – Scaling Law: Compute Optimal Curves on a Simple Model

*Abstract:* We  describe a program of analysis of stochastic gradient methods on high dimensional random objectives.  We illustrate some assumptions under which the loss curves are universal, in that they can completely be described in terms of some underlying covariances.   Furthermore, we give description of these loss curves that can be analyzed precisely.   We show how this can be applied to SGD on a simple power-law model.  This is a simple two-hyperparameter family of optimization problems, which displays 4 distinct phases of loss curves; these phases are determined by the relative complexities of the target, data distribution, and whether these are ‘high-dimensional’ or not (which in context can be precisely defined).  In each phase, we can also give, for a given compute budget, the optimal parameter dimensionality. Joint work with Elliot Paquette (McGill), Jeffrey Pennington (Google Deepmind), and Lechao Xiao  (Google Deepmind).


####	9:45 - 10:30	 Mert Gürbüzbalaban – Heavy Tail Phenomenon in Stochastic Gradient Descent

*Abstract*: Stochastic gradient descent (SGD) methods are workhorse methods for training machine learning models, particularly in deep learning. After presenting numerical evidence demonstrating that SGD iterates with constant step size can exhibit heavy-tailed behavior even when the data is light-tailed, in the first part of the talk, we delve into the theoretical origins of heavy tails in SGD iterations based on analyzing products of random matrices and their connection to various capacity and complexity notions proposed for characterizing SGD's generalization properties in deep learning. Key notions correlating with performance on unseen data include the 'flatness' of the local minimum found by SGD (related to the Hessian eigenvalues), the ratio of step size η to batch size b (controlling stochastic gradient noise magnitude), and the 'tail-index' (which measures the heaviness of the tails of the eigenspectra of the network weights). We argue that these seemingly disparate perspectives on generalization are deeply intertwined. Depending on the Hessian structure at the minimum and algorithm parameter choices, SGD iterates converge to a heavy-tailed stationary distribution. We rigorously prove this claim in linear regression, demonstrating heavy tails and infinite variance in iterates even in simple quadratic optimization with Gaussian data. We further analyze tail behavior with respect to algorithm parameters, dimension, and curvature, providing insights into SGD behavior in deep learning. Experimental validation on synthetic data and neural networks supports our theory. Additionally, we discuss generalizations to decentralized stochastic gradient algorithms and to other popular step size schedules including the cyclic step sizes. In the second part of the talk, we introduce a new class of initialization schemes for fully-connected neural networks that enhance SGD training performance by inducing a specific heavy-tailed behavior in stochastic gradients. Based on joint work with Yuanhan Hu, Umut Simsekli, and Lingjiong Zhu. 

####	10:30 - 10:50 Break		
	
####	10:50 - 11:35 Katya Scheinberg – Stochastic Oracles and Where to Find Them

*Abstract:* Majority of continuous optimization methods developed in the last decade, especially in application to ML training, are developed under the assumption that  approximate first order information is available to the method in some form. The assumption on the quality and reliability of this information can vary substantially from method to method. We will overview different methods of obtaining this information, including simple stochastic gradient via sampling, robust gradient estimation in adversarial settings, traditional and randomized finite difference methods and more. We will also consider second order and other related oracels. We will attempt to propose a somewhat unified definition of stochastic oracles, under which to compare what exists in the literature.	

####	11:35 - 12:20 Boris Hanin – Scaling Limits of Neural Networks

**Abstract:**  Large neural networks are often studied analytically through scaling limits: regimes in which taking some structural network parameters (e.g. depth, width, number of training datapoints, and so on) to infinity results in simplified models of network properties. I will survey several such approaches, starting with the NTK regime in which network width tends to infinity at fixed depth and dataset size. Here, networks are Gaussian processes at initialization and are equivalent to linear models (at least for regression tasks). While this regime is tractable, it precludes a study of feature learning. The deviation from this NTK regime is controlled at finite width by the depth-to-width ratio, which plays the role of the effective network depth. I will explain how this occurs and state several results on how this effective depth affects learning in neural networks. 
	
####	12:20 - 2:00	 Lunch and Discussion			

## Posters


* **Sinjini Banerjee** – *Measuring Training Variability Using Robust Statistics*
* **Zihan Wang** – *Implicit Bias of SGD in L2-regularized linear DNNs: One-way Jumps from High to Low Rank*
* **Archan Ray** – *Approximating Eigenvalues of Symmetric Matrices using Matrix-vector Query Algorithms*
* **Natalie Frank** – *Understanding Adversarial Risks*
* **Rishi Sonthalia** – *Error in Variables Regression: Benigh Overfitting, Covariate shifts, and Underparameterized Double Descent*
* **Lizzy Coda** – *Fiber Bundle Morphisms as a Framework for Modeling Many-to-Many Maps*
* **Max Vargas** – *Bias Introduced by Machine Processing*
* **Kai Tan** – *Uncertainty Quantification for Iterative Algorithms in Linear Models with Application to Early Stopping*
* **Ewerton Rocha Vieira** – *MORALS: Analysis of High-Dimensional Robot Controllers via Topological Tools in a Latent Space*
* **Jacob Zavatone-Veth** – *Scaling and Renormalization in High-dimensional Regression*
* **Mary Letey** – *Theory of In-Context Learning for Regression with Linear Attention*
* **Haixiao Wang** – *Unlocking Exact Recovery in Semi-Supervised Learning: Analysis of Spectral Method and Graph Convolution Network*
* **Alexander Atanasov** – *A Dynamical Model of Neural Scaling Laws*
