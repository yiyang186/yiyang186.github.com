---
title: 如何理解EM算法
date: 2017-05-06 10:16:00
categories:
  - 机器学习
tags: 
  - ESL
  - EM
---

介绍EM算法的材料里，我目前看过且觉得比较好的就是NG老师的CS229讲义和李航老师的统计学习方法。
我也提不出什么新东西，就结合混合高斯分布，在这两位牛人的基础上，谈一点自己觉得看待EM算法很重要的2个必须弄清楚的问题：为什么要有EM算法，为什么叫E步和M步，还解释了一些介绍EM算法时免不了要提到的公式。如果不把这些问题和公式解释清楚真的能理解em吗？我想可能不能

# 为什么要有EM算法

我把EM算法当做最大似然估计的拓展，解决难以给出解析解的最大似然估计（MLE）问题。
考虑高斯分布，它的最大似然估计是这样的：

{% raw %}$$\begin{aligned} \theta^*=\arg\max_\theta \sum_X \log L(\theta|X) \end{aligned}{\tag 1}$${% endraw %}

其中，{% raw %}$\theta =(\mu, \sigma), \theta^* =(\mu^*, \sigma^*), \log L(\theta|X) = \log P(X; \theta)${% endraw %}是对数似然函数，分号左边是随机变量，右边是模型参数。$P(X;\theta)$表示X的概率值函数，它是一个以$\theta$为参数的函数（很多人看不懂EM算法就是因为符号问题）。这里对$\theta$求导很容易解出$\theta^*$。


但如果这是个含有隐量Y的模型比如混合高斯模型，
{% raw %}$$P(X;\theta)=\sum_{k=1}^K\pi_kN(x; \mu_k, \sigma_k)=\sum_YP(Y;\pi)P(X|Y;\mu,\sigma){\tag 2}$${% endraw %}

上面假设共有K个高斯模型混合.每个高斯模型的参数为$\theta_k=(\mu_k, \sigma_k)$,每个高斯模型占总模型的比重为$\pi_k$。隐变量$Y \in \{y_1,y_2,...,y_K\}$表示样本$x_i$来自于哪一个高斯分布。分布列为：

Y|{% raw %}$y_1${% endraw %}|{% raw %}$y_2${% endraw %}|{% raw %}$y_3${% endraw %}|...
---|---|---|---|---
{% raw %}$p(y)${% endraw %}|{% raw %}$\pi_1${% endraw %}|{% raw %}$\pi_2${% endraw %}|{% raw %}$\pi_3${% endraw %}|...

可以认为，混合高斯分布的观测值是这样产生的：先以概率$\pi_k$抽取一个高斯分布$y_k$，再以该高斯分布$N(x;\mu_k, \sigma_k)$去生成观测x。其实这里的$\pi_k$ 就是Y的先验分布$P(Y;\pi)$ (这里特地加上； $\pi$ 表示P(Y)的参数是 $\pi$ ,你需要求出 $\pi$ 才能表示这个先验分布),而 $N(x; \mu_k, \sigma_k)$ 就是给定Y下的条件概率 $P(X|Y;\mu,\sigma)$ 
这时，令{% raw %}$\theta =(\mu, \sigma, \pi), \theta^* =(\mu^*, \sigma^*, \pi^*)${% endraw %}, 最大似然估计成了

{% raw %}
$$\begin{aligned} 
\theta^* &= \arg\max_\theta \sum_X \log P(X;\theta) \\
&=\arg\max_\theta \sum_X \log  \sum_YP(Y;\pi)P(X|Y;\mu,\sigma) \\
&=\arg\max_\theta \sum_X \log  \sum_YP(X,Y;\theta)
\end{aligned}\tag 3$$
{% endraw %}

据群众反映，求和、取对数、再求和，这种形式求偏导较为费劲（到底有多费劲。。。其实放在混合高斯这里也不是那么费劲，有的情形远比混合高斯复杂）要是能把\log 拿到求和的最里层就好了，直接对最里层的式子求偏导岂不快哉？于是就有了EM算法


# 为什么要分E步和M步


为了解决这个问题，有人想到了Jensen（琴生）不等式. $\log$ 是个凹函数，以隐变量Y的任一函数$f(Y)$举个例子：
{% raw %}$$\log E[f(Y)]=\log \sum_Y P(Y)f(Y) \geq \sum_Y P(Y)\log f(Y)=E[\log f(Y)]\tag 4$${% endraw %}
根据琴生不等式的性质，当随机变量函数 f(Y) 为常数时，不等式取等号。上式中的期望换成条件期望，分布 P(Y) 换成条件分布也是一样的。

注意(3)中的联合分布$P(X,Y;\theta)$在执行$\sum_Y$时可以把X看做是定值，此时我们可以把这个联合分布当做Y的随机变量函数（它除以P(Y)当然还是Y的随机变量函数）来考虑，并且引入一个关于Y的分布Q(Y)，具体是啥分布还不清楚,可能是给定某某的条件分布，只知道它是一个关于$\theta$的函数：

{% raw %}
$$\begin{aligned}
max &=\max_\theta \sum_X \log  \sum_YP(X,Y;\theta) \\
&=\max_\theta \sum_X \log  \sum_Y Q(Y;\theta) \cdot \frac{P(X,Y;\theta)}{Q(Y;\theta)} \\
&=\max_\theta \sum_X \log  E_Q[\frac{P(X,Y;\theta)}{P(Y;\theta)}] \\
&\geq \max_\theta \sum_X E_Q[\log  \frac{P(X,Y;\theta)}{Q(Y;\theta)}] \\
&= \max_\theta \sum_X \sum_Y Q(Y;\theta) \log  \frac{P(X,Y;\theta)}{Q(Y;\theta)}
\end{aligned}\tag 5$$
{% endraw %}

只有当
{% raw %}$$\frac{P(X,Y;\theta)}{Q(Y;\theta)}=c\tag 6$${% endraw %}
式(5)才能取等号，注意到Q是Y的某一分布，有$\sum_Y Q(Y;\theta)=1$这个性质，因此
{% raw %}
$$\begin{aligned}
Q(Y;\theta) &= \frac{P(X,Y;\theta)}{c} = \frac{P(X,Y;\theta)}{c \cdot \sum_Y Q(Y;\theta)} \\
&= \frac{P(X,Y;\theta)}{\sum_Y c \cdot Q(Y;\theta)} = \frac{P(X,Y;\theta)}{\sum_Y P(X,Y;\theta)} \\
&= \frac{P(X,Y;\theta)}{P(X;\theta)} = P(Y|X;\theta)
\end{aligned}\tag 7$$
{% endraw %}
所以只需要把Q取为给定X下，Y的后验分布，就能使式(5)取等号，下一步只需要最大化就行了.这时(5)为
{% raw %}$$\theta^* = \arg\max_\theta \sum_X \sum_Y P(Y|X;\theta) \log  \frac{P(X,Y;\theta)}{P(Y|X;\theta)}\tag 8$${% endraw %}

其中：
{% raw %}$$P(X,Y;\theta) = P(Y;\pi)P(X|Y;\mu,\sigma)= \pi_kN(x_i; \mu_k, \sigma_k)\tag 9$${% endraw %}
{% raw %}$$P(Y|X;\theta) = \frac{P(X,Y;\theta)}{\sum_Y P(X,Y;\theta)}= \frac{\pi_kN(x_i; \mu_k, \sigma_k)}{\sum_{k=1}^K \pi_kN(x_i; \mu_k, \sigma_k)}\tag{10}$${% endraw %}
好吧，直接对$(\mu, \sigma, \pi)$求导还是很麻烦，不过已经可以用迭代来最大化啦。

1）先根据式(10)，由$(\mu^{(j)}, \sigma^{(j)}, \pi^{(j)})$求后验概率
$$Q^{(j)}=P(Y|X;\theta^{(j)})$$

2）再把$Q^{(j)}$带入(8)中，

{% raw %}
$$\begin{aligned}
\theta^{(j+1)} &= \arg\max_\theta \sum_X \sum_Y Q^{(j)} \log  \frac{P(X,Y;\theta)}{Q^{(j)}} \\
&= \arg\max_\theta \sum_X \sum_Y (Q^{(j)} \log  P(X,Y;\theta)-Q^{(j)} \log  Q^{(j)}) \\
&= \arg\max_\theta \sum_X \sum_Y Q^{(j)} \log  P(X,Y;\theta)
\end{aligned}\tag{11}$$
{% endraw %}
就只需要最大化联合分布$P(X,Y;\theta)$了，最大化求出$(\mu^{(j+1)}, \sigma^{(j+1)}, \pi^{(j+1)})$后重复这2步。

M步很显然，就是最大化那一步，E步又从何谈起呢？式(11)可以写成

{% raw %}
$$\begin{aligned}
\theta^{(j+1)} &= \arg\max_\theta \sum_X \sum_Y Q^{(j)} \log  P(X,Y;\theta) \\
&= \arg\max_\theta \sum_X E_{Q^{(j)}} [\log  P(X,Y;\theta)] \\
&= \arg\max_\theta \sum_X E_{Y|X;\theta^{(j)}} [\log  P(X,Y;\theta)] \\
&= \arg\max_\theta \sum_X E_Y [\log  P(X,Y;\theta)|X;\theta^{(j)}]
\end{aligned}\tag{12}$$
{% endraw %}
其实，E步就是求给定X下的条件期望，也就是后验期望，使得式(5)的琴生不等式能够取等号，是对琴声不等式中,小的那一端进行放大，使其等于大的那一端，这是一次放大；M步最大化联合分布，通过0梯度，拉格朗日法等方法求极值点，又是一次放大。只要似然函数是有界的，只要M步中的0梯度点是极大值点，一直放大下去就能找到最终所求。

[我的知乎回答](https://www.yhihu.com/question/27976634/answer/163164402)
