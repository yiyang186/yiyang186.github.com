---
title: GBDT
tags:
---

# 回顾
我在另外2篇博客（{% post_link boosting与指数损失 boosting与指数损失 %}， {% post_link boosting与平方损失 boosting与平方损失 %}）中介绍了一些关于boosting的简单内容。在{% post_link boosting与平方损失 boosting与平方损失 %}最后借由平方损失和前向分步拟合算法推导出了梯度提升算法：

> 梯度提升算法
> 1. 初始化{% raw %}$f_0(x)=0${% endraw %}
> 2. 对于m=1到M:<br>(a)计算伪残差，即负梯度<br>{% raw %}$$r_{m-1}=-\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m-1}(x)}$${% endraw %} 若梯度接近与0，可提前结束迭代<br> (b)以$b(x;\gamma_m)$拟合伪残差,估计模型参数$\gamma_m$<br>(c)估计最优步长 <bt>{% raw %}$$\beta_m = \arg\min_{\beta}L(r_{m-1}, \beta b(x;\gamma_m)) = \arg\min_{\beta}L(y, f_{m-1}(x)+\beta b(x;\gamma_m)) $${% endraw %} (d)更新模型<br>{% raw %}$$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)$${% endraw %}

直到这里一直都没说用什么弱学习器$b(x;\gamma)$来拟合梯度（$-r_m$）,下面用决策树来拟合梯度，导出GBDT（Gradient Boosting Decision Tree,也叫 MART, Multiple Additive Regression Tree）算法。

# GBDT
从名字“梯度提升决策树”Gradient Boosting Decision Tree里可以看出，GBDT就是在梯度提升算法中用上决策树来做弱学习器$b(x;\gamma)$，下面我们不再用$b(x;\gamma)$这一表达式了，直接用决策树的表达式
$$T(x;\theta, R)=\sum_{j=1}^J\theta_jI(x\in R_j)$$
其中参数$R_j$为一个子特征空间，$\theta_j$为该空间上的常数预测值。

> 回顾一下决策树：决策树通过Gini系数或者信息增益求出每个特征维度的分裂点，再通过分裂点把特征空间分割成一个一个的矩形的子特征空间，在每个子空间内应用简单的分类或回归手段，如多数类、平均数，作为该子空间的预测值。<br>$$x \in R_j \Rightarrow T(x)=\theta_j$$
