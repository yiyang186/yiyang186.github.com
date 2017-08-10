---
title: 从梯度提升到GBDT
date: 2017-08-10 11:24:01
categories:
  - 机器学习
tags:
  - boosting
  - ensamble
  - 梯度提升
  - GBDT
  - 决策树
---

## 梯度提升(Gradient boosting)
对于任意的可微的损失函数，我们可以在迭代的过程中求出梯度，再以弱学习器去拟合负梯度来完成梯度提升算法。在{% post_link boosting与平方损失 boosting与平方损失 %}中，我们发现前向分步拟合算法每次迭代中，当前的残差就是经验损失相对于当前预测函数值的负梯度。需要注意:对于其他非平方损失的可微损失函数，负梯度不等于残差，而是残差的近似，因此在梯度提升算法中负梯度也被称为伪残差(pseudo-residuals)。
再次提一下：**当使用平方损失时，伪残差就是真残差，直接计算真残差就好了！但是梯度提升的适用范围更为广阔，所以我们没有提及用那种损失函数时，还是得说“伪残差”！！**
{% raw %}
$$r_{m}\approx -\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m}(x)} \tag{11}$$
{% endraw %}
这与{% post_link boosting与平方损失 boosting与平方损失 %}中的梯度提升算法相比，我们仍然是拟合真实的梯度，仍然是在拟合的梯度方向上下降。不同点是我们得到了一个好处：可以使用任意的可微的损失函数。此外，负梯度方向是当前迭代中的最速下降方向，弱学习器拟合负梯度(伪残差)能获得更快的收敛速度，在前向分步拟合的迭代中，弱学习器拟合残差是贪婪的，拟合梯度也是贪婪的，都是近似解，何不要收敛更快的方法？

> 梯度提升算法
> 1. 初始化{% raw %}$f_0(x)${% endraw %}为常量 <br>{% raw %}$$\gamma_0=\arg\min_{\gamma}L(y, b(x;\gamma))$$ $$f_0(x)=b(x;\gamma_0)$${% endraw %}
> 2. 对于m=1到M:<br>(a)计算伪残差，即负梯度<br>{% raw %}$$r_{m-1}=-\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m-1}(x)}$${% endraw %} 若梯度接近与0，可提前结束迭代<br> (b)以$b(x;\gamma_m)$拟合伪残差,估计模型参数$\gamma_m$<br>(c)估计最优步长 <bt>{% raw %}$$\beta_m = \arg\min_{\beta}L(r_{m-1}, \beta b(x;\gamma_m)) = \arg\min_{\beta}L(y, f_{m-1}(x)+\beta b(x;\gamma_m)) $${% endraw %} (d)更新模型<br>{% raw %}$$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)$${% endraw %}

<br>
> **注意**：这里的初始化与以往的前向分步拟合算法的初始化不同，其实只是把第二步中的第一次迭代放到初始化里来。此外，第一个弱学习器的权重为1。大家都有权重，其中一个权重设为1，相当于所有权重同乘以一个因子，对结果并没有影响。

直到这里一直都没说用什么弱学习器$b(x;\gamma)$来拟合梯度（$-r_m$）,下面用决策树来拟合梯度，导出GBDT（Gradient Boosting Decision Tree,也叫 MART, Multiple Additive Regression Tree）算法。

# GBDT
从名字“梯度提升决策树”Gradient Boosting Decision Tree里可以看出，GBDT就是在梯度提升算法中用上决策树来做弱学习器$b(x;\gamma)$，下面我们用T代替b来表示决策树
$$T(x;\gamma')=\sum_{j=1}^J\theta_jI(x\in R_j)$$
其中参数J为子空间的数量（即叶子节点），$\gamma'=\{R_j, \theta_j\}_1^J$, $R_j$为一个子特征空间，$\theta_j$为该空间上的常数预测值。

> 回顾一下决策树：
> 决策树通过Gini系数或者信息增益求出每个特征维度的分裂点，再通过分裂点把特征空间分割成一个一个的矩形的子特征空间，在每个子空间内应用简单的分类或回归手段，如多数类、平均数，作为该子空间的预测值。<br>$$x \in R_j \Rightarrow T(x)=\theta_j$$

提升树模型是这样的树加权和{% raw %}
$$\begin{aligned}
f(x) &=\sum_{m=1}^M\beta_m'T(x;\gamma_m')\\
&=\sum_{m=1}^M\beta_m'\sum_{j=1}^{J_m}\theta_{mj}I(x\in R_{mj})\\
&=\sum_{m=1}^M\sum_{j=1}^{J_m}\beta_m'\theta_{mj}I(x\in R_{mj})\\
&=\sum_{m=1}^M\sum_{j=1}^{J_m}\beta_{mj}I(x\in R_{mj})
\end{aligned} \tag2$$
{% endraw %}
这里令$\beta_{mj}=\beta_m'\theta_{mj}$，这样相当于把树内的子空间当做弱学习器，预测值为样本是否在该子空间内$I(x\in R_{mj})$，而权重为子空间的预测值$\theta_{mj}$与该树权重$\beta_m'$的乘积$\beta_{mj}=\beta_m'\theta_{mj}$。更简单地，令$\gamma_m=\{R_{mj}, \beta_{mj}\}_{j=1}^{J_m}$，式(2)也相当于学习不带权重参数的M棵树：
$$f(x) =\sum_{m=1}^M\sum_{j=1}^{J_m}\beta_{mj}I(x\in R_{mj})=\sum_{m=1}^MT(x;\gamma_m)$$
这里每棵树内每个子空间的预测值就已经包含了该树在最终委员会模型中的权重信息，可以减少要学习的参数，加快学习速度。那么，GBDT每一步迭代的优化就成了{% raw %}
$$\begin{aligned}\gamma_m
&=\arg\min_{\gamma}\sum_{i=1}^ML\bigg(y_i, f_{m-1}(x_i)+T(x_i;\gamma_m)\bigg)\\
&=\arg\min_{\beta, R}\sum_{i=1}^ML\bigg(y_i, f_{m-1}(x_i)+\sum_{j=1}^{J_m}\beta_{mj}I(x\in R_{mj})\bigg)\\
\end{aligned}\tag3$$
{% endraw %}
根据{% post_link boosting与平方损失 boosting与平方损失 %}中介绍的对任意可微损失函数的梯度提升算法，我们只需用$T(x;\gamma_m)$去拟合当前迭代中的伪残差即可{% raw %}
$$(\beta_m, R_m)=\arg\min_{\beta, R}\sum_{i=1}^ML(-\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m-1}(x)}, \sum_{j=1}^{J_m}\beta_{mj}I(x\in R_{mj}))\tag4$$
{% endraw %}

# 总结
从{% post_link boosting与指数损失 boosting与指数损失——Adaboost %}到{% post_link boosting与平方损失 boosting与平方损失 %}，再到这篇博客，大致总结一下GBDT的由来：
1. 想把多个弱学习器组通过加权求和组成“不平等的委员会”，是一种基于学习器的加法模型{% raw %}
$$f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m)$$ {% endraw %}
2. 使用贪心的前向分步拟合算法，迭代地求解弱学习器，求解新的弱学习器时不改变已经求解好的弱学习器{% raw %}
$$(\beta_m, \gamma_m)=\arg\min_{\beta, \gamma}\sum_{i=1}^NL\bigg(y_i, f_{m-1}(x_i)+\beta b(x_i;\gamma)\bigg)$$ 
{% endraw %}
3. 迭代中，由于加法模型的性质，求解新的弱学习器实际上是拟合当前残差{% raw %}
$$\begin{aligned}
&\min L\bigg(y, f_{m-1}(x)+\beta_m b(x;\gamma_m)\bigg) \\
= &\min L\bigg(y-f_{m-1}(x), f_{m-1}(x)+\beta_m b(x;\gamma_m)-f_{m-1}(x)\bigg) \\
= &\min L\bigg(r_{m-1}, \beta_m b(x;\gamma_m)\bigg)
\end{aligned}$$ 
{% endraw %}
4. 梯度提升算法是在前向分步拟合的每次迭代中，找到经验损失相对于预测函数值的最速下降方向（负梯度方向），用弱学习器拟合负梯度，最终使梯度减少到接近0。当取平方经验损失时，当前残差就是当前负梯度。梯度提升算法还适用于别的损失函数，当前负梯度可以看做当前残差的近似，称为伪残差。而且负梯度方向是当前迭代中的最速下降方向，弱学习器拟合负梯度(伪残差)能获得更快的收敛速度。{% raw %}
$$r_{m-1} \approx-\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m-1}(x)}$$
{% endraw %}
5. 使用决策树作为弱学习的梯度提升算法是GBDT
