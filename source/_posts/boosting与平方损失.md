---
title: boosting与平方损失——梯度提升
date: 2017-08-06 21:06:22
categories:
  - 机器学习
tags:
  - boosting
  - 损失函数
  - ensamble
  - 梯度提升
  - 梯度下降
---

# Boosting
在另一篇博客中({% post_link boosting与指数损失 boosting与指数损失 %})介绍了关于Boosting的一些简单内容。
- boosting的过程是不断地修改数据集，在此之上应用弱学习器，由此产生一个弱学习器序列{% raw %}$b(x;\gamma_m), m = 1, 2, ..., M${% endraw %}。最后通过加权的多数表决来合并每个弱学习器的预测结果。
- Boosting是一种建立在弱学习器集合上的加法模型。
- 通过前向分步拟合的方法来拟合Boosting，相继添加新的弱学习器到Boosting“委员会”里，而不调整已添加的弱学习器的模型参数及其在委员会中的权重系数，这是一种贪心的方法，一次只拟合一个最优的弱学习器。
- 前向分步拟合方法使用指数损失拟合Boosting分类模型，等价于Adaboost

- > 前向分步拟合算法
  > 1. 初始化{% raw %}$f_0(x)=0${% endraw %}
  > 2. 对于m=1到M:<br>(a)计算 <bt>{% raw %}$$(\beta_m, \gamma_m)=\arg\min_{\beta, \gamma}\sum_{i=1}^NL\bigg(y_i, f_{m-1}(x_i)+\beta b(x_i;\gamma)\bigg)$${% endraw %} (b)更新{% raw %}$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)${% endraw %}

# 平方损失+前向分步拟合=梯度提升(Gradient boosting)
## 平方经验损失的梯度
令f为所求的boosting模型
{% raw %}
{% raw %}$$f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m) \tag 1$${% endraw %}
{% endraw %}

面对回归问题，通常使用平方损失：
{% raw %}
$$L(y, f(x))=(y-f(x))^2\tag 2$$
{% endraw %}
令{% raw %}$y=(y_1,...,y_N)^T, f(x)=(f(x_1),...,f(x_N))^T${% endraw %}为boosting模型的预测值，那么最小化平方经验损失的过程可以等价于
{% raw %}
$$
\min_fL(y, f(x))=\min_f(y-f(x))^T(y-f(x))=\min_f\frac{1}{2}(y-f(x))^T(y-f(x))
\tag 3$$
{% endraw %}

显然这是凸优化，其负梯度方向为
{% raw %}
$$\begin{aligned}
- \frac{\partial}{\partial f(x)} \bigg(\frac{1}{2}(y-f(x))^T(y-f(x))\bigg)
&= - \frac{1}{2} \cdot 2 \cdot \Big(\frac{\partial}{\partial f(x)} (y-f(x))^T\Big) \cdot (y-f(x)) \\
&= - \Big(\frac{\partial y^T}{\partial f(x)} - \frac{\partial f^T(x)}{\partial f(x)}\Big) \cdot (y-f(x)) \\
&= -(0-I)\cdot (y-f(x)) \\
&= y-f(x)
\end{aligned} \tag4$$
{% endraw %}
我们发现，模型残差就是平方经验损失相对于模型预测值的负梯度方向。不过在模型$f$没有求出来之前，我们没法计算梯度。
> **注意**： 这和我们平时所见的梯度不同，一般梯度下降法里的梯度是经验损失相对于模型参数的梯度，而这里的梯度是经验损失相对于模型预测值的梯度。

## 前向分步拟合最小化平方经验损失
在前向分步拟合算法中，第m次迭代的模型为
{% raw %}
{% raw %}$$f_m(x) = \sum_{i=1}^m \beta_i b(x; \gamma_i)=f_{m-1}+\beta_m b(x;\gamma_m)\tag 5$${% endraw %}
{% endraw %}
每次迭代，最小化平方经验损失的过程为
{% raw %}
$$\begin{aligned}
\min_{f_m}L(y, f_m(x))
&= \min_{f_m}(y-f_m(x))^T(y-f_m(x)) \\
&= \min_{f_m}\|y-f_m(x)\|^2\\
&= \min_{\beta, \gamma}\|y-f_{m-1}(x)-\beta_mb(x;\gamma_m)\|^2\\
&= \min_{\beta, \gamma}\|r_{m-1} - \beta_m b(x;\gamma)\|^2
\end{aligned}
\tag{6}$$
{% endraw %}
令{% raw %}$r_{m}=(r_{m1},...,r_{mN})^T, r_{mi}=y_i - f_m(x_i)${% endraw %}，当然{% raw %}$r_{m-1}${% endraw %}就是是当前模型(第m-1次迭代所产生的模型)的残差。这样，对于平方损失，每一次迭代都是把对当前模型残差拟合的最好的弱分类器及其系数{% raw %}$\beta_mb(x;\gamma_m)${% endraw %}加到新模型{% raw %}$f_m(x)${% endraw %}里。将这个目标函数带入到前向分布拟合算法中来

> 平方损失的前向分步拟合算法
> 1. 初始化{% raw %}$f_0(x)=0${% endraw %}
> 2. 对于m=1到M:<br>(a)计算残差<br>{% raw %}$$r_{m-1}=y - f_{m-1}(x)$${% endraw %} (b)估计模型参数，拟合残差 <bt>{% raw %}$$(\beta_m, \gamma_m)=\min_{\beta, \gamma}\|r_{m-1} - \beta b(x;\gamma)\|^2 \tag 7$${% endraw %} (c)更新{% raw %}$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)${% endraw %}

式(7)中$\beta, \gamma$的优化不存在相互依赖，可以分开优化，式(7)等价于
{% raw %}
$$\bigg\{ \begin{matrix}
\gamma_m = \arg\min_{\gamma}\|r_{m-1} - b(x;\gamma)\|^2 \\
\beta_m = \arg\min_{\beta}\|r_{m-1} - \beta b(x;\gamma)\|^2
\end{matrix}
\tag{8}$$
{% endraw %}


## 平方损失下的梯度提升
由上一小节我们知道{% raw %}$r_{m-1}${% endraw %}就是平方经验损失相对于当前模型预测值的负梯度方向，那么这里的{% raw %}$b(x;\gamma_m)${% endraw %}就是对当前模型预测值的负梯度方向最优的拟合。而一维实数{% raw %}$\beta_m${% endraw %}可以看做是梯度下降时的步长。上述算法等价于

> 平方损失下的梯度提升
> 1. 初始化{% raw %}$f_0(x)=0${% endraw %}
> 2. 对于m=1到M:<br>(a)计算残差<br>{% raw %}$$r_{m-1}=y - f_{m-1}(x)$${% endraw %} 则梯度为{% raw %}$-r_{m-1}${% endraw %}, 若梯度非常接近0可提前结束迭代<br> (b)估计模型参数，拟合残差 <bt>{% raw %}$$\gamma_m = \arg\min_{\gamma}\|r_{m-1} - b(x;\gamma)\|^2 \tag 9$${% endraw %}(c)估计最优步长 <bt>{% raw %}$$\beta_m = \arg\min_{\beta}\|r_{m-1} - \beta b(x;\gamma_m)\|^2 \tag{10}$${% endraw %} (d)更新模型，梯度下降<br>{% raw %}$$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)=f_{m-1}(x)-\beta_m (-b(x; \gamma_m))$${% endraw %}

考虑实际的梯度下降算法
> 在梯度下降算法中，每次迭代需要求当前搜索位置的梯度，然后沿着负梯度方向搜索，直到找到0梯度位置为止。

与梯度下降算法比较，平方损失下的梯度提升算法，相当于以模型预测值$f(x)$为参数做梯度下降。这里的残差为真实的负梯度，我们以弱学习器去拟合真实的负梯度，再沿着拟合的负梯度方向搜索。实际上并不是在真实的梯度方向下降，而是在拟合的梯度方向下降。

## 梯度提升(Gradient boosting)
对于任意的可微的损失函数，我们可以在迭代的过程中求出梯度，并以负梯度作为残差的近似，再以弱学习器去拟合负梯度来完成梯度提升算法。但是需要注意，对于其他非平方损失的可微损失函数，负梯度不等于残差，而是残差的近似，因此负梯度也被称为伪残差(pseudo-residuals)。
{% raw %}
$$r_{m}\approx -\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m}(x)} \tag{11}$$
{% endraw %}
与原来相比，我们得到一个好处，即可以使用任意的可微的损失函数，且仍然是拟合真实的梯度，仍然是在拟合的梯度方向上下降；坏处是前向分步拟合算法中的残差不在等于负梯度。不过前向分步拟合算法本来就是贪心的迭代算法，只要我们每一步迭代都能尽量地使残差{% raw %}$r_{m}${% endraw %}减小，并不违反前向分步拟合算分的初衷。最后，适用于任意可微损失函数的梯度提升算法为

> 梯度提升算法
> 1. 初始化{% raw %}$f_0(x)=0${% endraw %}
> 2. 对于m=1到M:<br>(a)计算伪残差，即负梯度<br>{% raw %}$$r_{m-1}=-\bigg[\frac{\partial L(y, f(x))}{\partial f(x)}\bigg]_{f(x)=f_{m-1}(x)}$${% endraw %} 若梯度接近与0，可提前结束迭代<br> (b)以$b(x;\gamma_m)$拟合伪残差,估计模型参数$\gamma_m$<br>(c)估计最优步长 <bt>{% raw %}$$\beta_m = \arg\min_{\beta}L(r_{m-1}, \beta b(x;\gamma_m)) = \arg\min_{\beta}L(y, f_{m-1}(x)+\beta b(x;\gamma_m)) $${% endraw %} (d)更新模型<br>{% raw %}$$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)$${% endraw %}


<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: 'boosting_square_loss',
  title: 'boosting与平方损失——梯度提升',
  owner: 'yiyang186',
  repo: 'blog_comment',
  oauth: {
    client_id: '2786ddc8538588bfc0c8',
    client_secret: '83713f049f4b7296d27fe579a30cdfe9e2e45215',
  },
})
gitment.render('container')
</script>