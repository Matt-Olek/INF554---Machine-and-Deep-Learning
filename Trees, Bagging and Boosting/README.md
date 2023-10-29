---
tags:
  - Informatique
Thèmes:
  - "[[Machine Learning]]"
  - "[[Decision Tree Classsification]]"
  - "[[Bagging Trees]]"
Avec: 
Date de création: 2023-10-27
---
## A - Trees 

### Question 1 

- The first property **($\Phi$ achieves its unique maximum at point $\left(\dfrac{1}{K},\ldots,\dfrac{1}{K}\right)$)** means that when the proportion of samples is the same for every class, the impurity is maximum because we have not succeeded to purify anything, we cannot distinct any classes from each other. It is unique because it is the only configuration where every class has the same number of samples.
- On the contrary, the property ($\Phi$ achieves its minima exclusively at the points $\left(1,0,\ldots,0\right)$, $\left(0,1,0,\ldots,0\right)$, ..., $\left(0,\ldots,0,1\right)$) means that if we only get one type of class in our batch, it means that reached full purity, which results in the minimum of impurity.
- Finally, the last property **($\Phi$ is a symmetric function of $p_1,\ldots,p_K$, i.e., if there is a permutation of the variables $p_k$, the function's output will remain unchanged.)** underlines the fact that it is equal in terms of purity to get a clear advantage in one class than to get the same advantage in any other one. We do not specify that any class is more important than another to sort.


Now Let's prove that the Gini index satisfies theses properties for $K=2$ :
For $K=2$ : $$i(t) = 1-p_1^2(t) - p_2^2(t)$$
But : $$p_2 = 1-p_1$$
So : $$i(t) = 1-p_1^2(t) - (1-p_1(t))^2 = 2p_1(t)(1-p_1(t)) $$
Which is a 2nd degree polynomial function in p_1 with single max at :$$p_1(t) = \frac{1}{2} = \frac{1}{K} = p_2(t)$$
- Which **satisfies the 1st property**.
- The second property is directly satisfied by the fact that $f:p_1\to 2p_1(t)(1-p_1(t))$ strictly increases on $[0:\frac{1}{2}]$ and strictly decreasing on $[\frac{1}{2}:1]$ which implies that the minimum is only achieved in $p_1=0$ (and $p_2=1$) or $p_1=1$ (and $p_2=0$) but $f(0) = f(1) = 0$. So the minima is exclusively achieved at the points (0,1) and (1,0). That **satisfies the 2nd property**.
- The **last property** is satisfied because of the fact that $f:p_1\to 2p_1(t)(1-p_1(t))$ is **symmetric** with the axis $p_1=\frac{1}2$ and $p_2 = 1-p_1$. A permutation of $p_1$ and $p_2$ will not change the output of the Gini index function.

We proved that the Gini index is an Impurity function for $K=2$ .

## B - Bagging Trees

### Question 2 

Probability that a given observation in a data set of size N is part of a bootstrap sample is $$1-\left(1-\frac{1}{N}\right)^N$$
**Proof** : 
***
- The probability that the **first** bootstrap observation is not the $i-th$ observation from the original sample of size N is $$P_i = 1−\frac{1}N$$
- It is the same probability for the **second bootstrap** etc..
- So the probability that the $i-th$ observation is not in the bootstrap sample is $$P_i^N = (1−\frac{1}N)^N$$
- Finally the probability that a given observation in a data set of size N is part of a bootstrap sample is $$1-\left(1-\frac{1}{N}\right)^N$$
***
 The limit of this probability when $N \to +\infty$ is computed with equivalents  : 
$$\left(1-\frac{1}{N}\right)^N = e^{N\ln(1-\frac{1}{N})}$$

Besides when $N \to +\infty$ :  : 
$$\ln(1-\frac{1}{N})\sim -\frac{1}{N}$$ so : $$N\ln(1-\frac{1}{N})\sim -N\frac{1}{N} = -1$$
This way we get when $N \to +\infty$ : $$1-\left(1-\frac{1}{N}\right)^N \to 1 - e^{-1}\approx 0.63$$ 

***


### Question  3

The most important predictor according to MDA seems to be the **plasma glucose concentration**. It is consistent with real life as glucose is highly affecting diabetes patients.
On top of that, it is consistent with the resulting coefficients of the logistic regression from task 5 : The **coefficient of Glucose** is **the highest**, even in absolute values, which implies that it is the one with the **most important impact** on the final decision. Moreover, it is **positive** as more glucose implies a greater chance of having diabetes.

## C - Boosting Trees

### Question 4

We can observe that some values are getting a way too important weight compared to the other values, they may come from low quality data and AdaBoost seems to be really **sensitive to outsiders and low quality data**.

### Question 5

An idea to solve this sensitivity would be to **make Adaboost less sensitive** in the beginning of its training by choosing **another initial weight distribution** than uniform. My idea was to create a weight distribution based on the **distance to the center of mass** of each class. A value's weight is higher as it is close to the center of mass and the "cloud" of points. It is interesting to use this distribution in a "bag" distribution of the values but it may be inefficient in an other configuration with more original shapes.

We use the distribution of weights : $$wi = \log(\frac{max_{distance}}{distance_i(center\space of \space mass)})$$and then normalize it. 

That process **decreases the weight of outsiders** in the initial state.
***
However if we wanted to keep such a distribution in every iterations, we could modify our step 2.d by setting $w_i$ to $w_i exp[\frac{\alpha_b I (y_i \neq C_b(x_i))}{distance_i(center\space of \space mass)}], i = 1,\ldots,N$.

### Question 6

**XGBoost** :
- Based on gradient boosting trees
- Powerful and flexible
- Easily scalable and efficient for large datasets
- Prevents overfitting with regularization techniques

**LightGBM** : 
- Based on High-performance gradient boosting trees
- Faster and more efficient than XGBoost
- Leaf-wise tree growing strategy to reduce training time and improve accuracy
- Histogram-based feature and cache optimization

In this **particular** dataset, XGBoost seems to be more efficient than LightGBM as their respective accuracy are : **0,823 and 0,797**
However, they are really close and way higher than the AdaBoost results.
