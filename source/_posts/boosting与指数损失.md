---
title: boosting与指数损失——Adaboost
date: 2017-08-01 15:59:22
categories:
  - 机器学习
tags:
  - boosting
  - 损失函数
  - ensamble
  - adaboost
---

# boosting-不平等的委员会
boosting提升方法的动机是合并许多“弱”学习器输出以产生有效的“委员会”。从这一角度看，boosting与同为ensamble集成方法的bagging袋装方法非常相似。其实这种相似是非常表面的，boosting与bagging有着本质的区别。

boosting的过程就是不断地修改数据集，在此之上应用弱学习器，由此产生一个弱学习器序列{% raw %}$b(x;\gamma_m), m = 1, 2, ..., M${% endraw %}。最后通过加权的多数表决来合并每个弱学习器的预测结果。总的学习器可以写成：
{% raw %}
$$f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m) \tag 1$$
{% endraw %}
其中，{% raw %}$\beta_m > 0${% endraw %}为若学习器{% raw %}$b(x;\gamma_m)${% endraw %}的权重，{% raw %}$\gamma_m${% endraw %}为{% raw %}$b_m(x)${% endraw %}的模型参数。显然，Boosting是加权表决，和Bagging的平均表决有着本质区别：前者的委员会是不平等的，后者的委员会是一人一票的。这也导致二者修改数据集的方式不相同。从这里开始，Boosting和Bagging彻底不同了。

# Boosting与加法模型
观测式(1),可以发现Boosting是一种建立在弱学习器集合上的加法模型。
{% raw %}
$$f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m)$$
{% endraw %}
> - 在单隐藏层神经网络中，{% raw %}$b(x;w_m,b_m)=\sigma(w_m^Tx+b_m)${% endraw %}，其中{% raw %}$\gamma_m=(w_m,b_m)${% endraw %}是偏置和连接上的权重<br>
> - 在小波变换里，{% raw %}$b(x;j_m,k_m)=2^{j_m/2}\phi(2^{j_m}x-k_m)${% endraw %}, 其中{% raw %}$\gamma_m=(j_m,k_m)${% endraw %}是对母小波{% raw %}$\phi${% endraw %}(如Haar，symmlet)的缩放和位移
> - 在MARS多元自适应回归样条里,{% raw %}$b_m(x;t_m)=b_i(x)(x-t_m)_+ + b_i(x)(t_m-x)_+, i \leq m${% endraw %}, 其中{% raw %}$\gamma_m=t_m${% endraw %}是分段函数的扭结
> - 对于树，{% raw %}$b(x;R_{mj},\theta_m)=\sum_{j=1}^J\theta_{mj}I(x \in R_{mj})${% endraw %}，其中{% raw %}$\gamma_m=(R_{mj},\theta_{mj})${% endraw %}是树的矩形子空间划分和每个子空间上的均值(回归)或多数类(分类)

模型求解策略一般都是最小化经验损失，如
{% raw %}
$$\min_{\beta_m, \gamma_m}\sum_{i=1}^NL\bigg(y_i, \sum_{m=1}^M\beta_m b(x_i;\gamma_m)\bigg)\tag 2$$
{% endraw %}
但是数值求解boosting“委员会”时，“委员会”中还有多个弱学习器，所有弱学习器一同求解是比较困难的(当时是90年代)，但是贪心地，迭代地，一次只拟合一个弱学习器则是可行且快速的解决方案。

# 前向分步拟合Boosting
通过相继添加新的弱学习器到“委员会”（总的模型）里，而不调整已添加的弱学习器的模型参数及其在委员会中的权重系数，来逼近式(2)的解。算法如下

> 前向分步拟合算法
> 1. 初始化{% raw %}$f_0(x)=0${% endraw %}
> 2. 对于m=1到M:<br>(a)计算 <bt>{% raw %}$$(\beta_m, \gamma_m)=\arg\min_{\beta, \gamma}\sum_{i=1}^NL\bigg(y_i, f_{m-1}(x_i)+\beta b(x_i;\gamma)\bigg) \tag 3$${% endraw %} <br>(b)更新{% raw %}$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)${% endraw %}

算法的要点是：在第m次迭代中，求解最优的弱学习器{% raw %}$b(x;\gamma_m)${% endraw %}和响应的系数{% raw %}$\beta_m${% endraw %},并将其添加到当前的“委员会”{% raw %}$f_{m-1}(x)${% endraw %}中，由此产生新的“委员会”{% raw %}$f_{m}(x)${% endraw %}。之前添加的项并不会改变，每次都添加最优的弱学习器，这包含着**贪心**的思想。

# 前向分步拟合+指数损失=Adaboost

## 指数经验损失
考虑一个2分类问题，数据集为{% raw %}$\{x_i, y_i\}_{i=1}^N${% endraw %}，其中 {% raw %}$y_i \in \{ -1, 1\}${% endraw %}。面对分类问题，有时我们会使用指数损失：
{% raw %}$$L(y,f(x))=\exp(-yf(x))$${% endraw %}
对式(3)我们使用指数损失：

{% raw %}
$$\begin{aligned}(\beta_m, \gamma_m) 
    &= \arg\min_{\beta, \gamma}\sum_{i=1}^N \exp[-y_i(f_{m-1}(x_i)+\beta b(x_i;\gamma))] \\
    &= \arg\min_{\beta, \gamma}\sum_{i=1}^N \exp(-y_if_{m-1}(x_i))\exp(-y_i\beta b(x_i;\gamma)) \\
    &= \arg\min_{\beta, \gamma}\sum_{i=1}^N w_i^{(m)}\exp(-y_i\beta b(x_i;\gamma))
\end{aligned}\tag 4 $$
{% endraw %}

由于{% raw %}$\exp(-y_if_{m-1}(x_i))${% endraw %}和{% raw %}$\beta, \gamma${% endraw %}无关，可以看做数值优化求解过程中的常数，但它与i,m有关，即和样本、迭代次数有关，令
{% raw %}$$w_i^{(m)}=\exp(-y_if_{m-1}(x_i)) \tag 5$${% endraw %}
把它看做第m次迭代时，样本i的损失权重，样本权重根据迭代而改变。

## 求解弱学习器参数
{% raw %}$\beta${% endraw %}与{% raw %}$\gamma${% endraw %}不存在相互依赖的关系，可以分开优化，对于任意的{% raw %}$\beta > 0${% endraw %}与样本无关,因此

{% raw %}
$$\begin{aligned}
\gamma_m & = \arg\min_{\gamma}\sum_{i=1}^N w_i^{(m)}\exp(-y_i\beta b(x_i;\gamma)) \\
& = \arg\min_{\gamma}\exp(\beta)\sum_{i=1}^N w_i^{(m)}\exp(-y_i b(x_i;\gamma)) \\
& = \arg\min_{\gamma}\sum_{i=1}^N w_i^{(m)}\exp(-y_i b(x_i;\gamma)) \\
& = \arg\min_{\gamma}\sum_{i=1}^N w_i^{(m)}I(y_i \neq b(x_i;\gamma))
\end{aligned} \tag 6 $$
{% endraw %}

> 注意{% raw %}$y_i${% endraw %}的取值为{+1，-1}，当{% raw %}$y_i=b(x;\gamma)${% endraw %}时，{% raw %}$\exp(-y_i b(x_i;\gamma))${% endraw %}和{% raw %}$I(y_i \neq b(x_i;\gamma))${% endraw %}都取到各自的最小值;当{% raw %}$y_i \neq b(x;\gamma)${% endraw %}时，{% raw %}$\exp(-y_i b(x_i;\gamma))${% endraw %}和{% raw %}$I(y_i \neq b(x_i;\gamma))${% endraw %}都取到各自的最大值。


## 求解弱学习器在boosting中的系数
对{% raw %}$\gamma${% endraw %}的优化变成了对单弱学习器参数的求解，得到{% raw %}$\gamma_m${% endraw %}。下面优化系数{% raw %}$\beta_m${% endraw %}。

{% raw %}
$$\begin{aligned}
\beta_m & = \arg\min_{\beta}\sum_{i=1}^N w_i^{(m)}\exp(-y_i\beta b(x_i;\gamma_m)) \\
& = \arg\min_{\beta}\sum_{i=1}^N \bigg\{ \begin{matrix}w_i^{(m)}\exp(-\beta) & y_i = b(x; \gamma_m) \\ w_i^{(m)}\exp(\beta) & y_i \neq b(x; \gamma_m)\end{matrix} \\
& = \arg\min_{\beta}\sum_{y_i = b(x; \gamma_m)} w_i^{(m)}e^{-\beta} + \sum_{y_i \neq b(x; \gamma_m)} w_i^{(m)}e^{\beta}\\
& = \arg\min_{\beta} \bigg(e^{-\beta}\sum_{y_i = b(x; \gamma_m)} w_i^{(m)} + e^{\beta}\sum_{y_i \neq b(x; \gamma_m)} w_i^{(m)}\bigg) \\
& = \arg\min_{\beta} \bigg(e^{-\beta}\sum_{y_i = b(x; \gamma_m)} w_i^{(m)} + e^{-\beta}\sum_{y_i \neq b(x; \gamma_m)} w_i^{(m)} - e^{-\beta}\sum_{y_i \neq b(x; \gamma_m)} w_i^{(m)} + e^{\beta}\sum_{y_i \neq b(x; \gamma_m)} w_i^{(m)}\bigg) \\
& = \arg\min_{\beta} \bigg(e^{-\beta}\sum_{i=1}^N w_i^{(m)} + (e^{\beta}-e^{-\beta})\sum_{y_i \neq b(x; \gamma_m)} w_i^{(m)} \bigg)\\
& = \arg\min_{\beta} \bigg(e^{-\beta}\sum_{i=1}^N w_i^{(m)} + (e^{\beta}-e^{-\beta})\sum_{i=1}^N w_i^{(m)}I(y_i \neq b(x; \gamma_m))\bigg)
\end{aligned}\tag 7$$
{% endraw %}

式(7)可以看做是{% raw %}$\beta${% endraw %}的一元函数优化问题，由于{% raw %}$e^{-\beta}${% endraw %}与{% raw %}$e^{\beta}${% endraw %}都是凸函数，它们的组合也是凸函数，因此0梯度点是全局最优解。令目标函数导数为0

{% raw %}
$$\begin{aligned}
& \frac{\partial}{\partial \beta} \bigg(e^{-\beta}\sum_{i=1}^N w_i^{(m)} + (e^{\beta}-e^{-\beta})\sum_{i=1}^N w_i^{(m)}I(y_i \neq b(x; \gamma_m))\bigg) \\
& = -e^{\beta}\sum_{i=1}^N w_i^{(m)} + 2e^{-\beta}\sum_{i=1}^N w_i^{(m)}I(y_i \neq b(x; \gamma_m)) = 0
\end{aligned} \tag 8$$
{% endraw %}

解式(8)得
{% raw %}$$\beta_m=\frac{1}{2}\log\frac{1-err_m}{err_m} \tag 9$${% endraw %}
其中
{% raw %}$$err_m=\frac{\sum_{i=1}^Nw_i^{(m)}I(y_i \neq b(x; \gamma_m))}{\sum_{i=1}^Nw_i^{(m)}} \tag{10}$${% endraw %}
是最小化的、当前迭代中的、单个弱学习器的、带权的误差率。这样我们可以更新“委员会”了：
{% raw %}$$f_m(x)=f_{m-1}(x)+\beta_m b(x; \gamma_m)$${% endraw %}
别忘了前面定义的样本损失权重式(5),它与迭代次数m有关，每次迭代都需要更新

{% raw %}
$$\begin{aligned}
w_i^{(m+1)} & = \exp(-y_if_m(x_i)) \\
& = \exp \Big(-y_i(f_{m-1}(x_i)+\beta_m b(x_i; \gamma_m))\Big) \\
& = \exp(-y_if_{m-1}(x_i)) \cdot \exp(-y_i\beta_m b(x_i; \gamma_m)) \\
& = w_i^{(m)} \cdot \exp(-\beta_m y_i b(x_i; \gamma_m)) \\
& = w_i^{(m)} \cdot \exp \Big(\beta_m \cdot (2I(y_i \neq b(x_i; \gamma_m))-1)\Big) \\
& = w_i^{(m)} \cdot \exp \Big(2\beta_m I(y_i \neq b(x_i; \gamma_m))-\beta_m \Big) \\
& = w_i^{(m)} \cdot \exp \Big(2\beta_m I(y_i \neq b(x_i; \gamma_m))\Big)e^{-\beta_m}
\end{aligned}  \tag{11}$$
{% endraw %}

## 导出与adaboost的等价性
因为每个样本的权重都乘以因子{% raw %}$e^{-\beta_m}${% endraw %},所以乘不乘都没关系，令
{% raw %}$$\alpha_m = 2\beta_m = \log\frac{1-err_m}{err_m} \tag{12}$${% endraw %}
有
{% raw %}$$w_i^{(m+1)} = w_i^{(m)} \cdot \exp \Big(\alpha_m I(y_i \neq b(x_i; \gamma_m))\Big) \tag{13}$${% endraw %}
最后，每个弱学习器的系数都乘以2，学习器间的权重分布未变，总模型“委员会”为
{% raw %}$$f(x)=sign\Big( \sum_{m=1}^M \alpha_m b_m(x)\Big)$${% endraw %}

我们把上面的过程简化，写成算法的形式：
> 1. 初始化样本权重{% raw %}$w_i=1/N,i=1,2,...,N${% endraw %}<br>
> 2. 对于m=1,...,M:<br>
> (a)在样本权重为{% raw %}$w_i${% endraw %}的训练集上拟合一个弱学习器{% raw %}$b(x;\gamma_m)${% endraw %}<br>(b)计算<br>{% raw %}$$err_m=\frac{\sum_{i=1}^Nw_i^{(m)}I(y_i \neq b(x; \gamma_m))}{\sum_{i=1}^Nw_i^{(m)}}$${% endraw %} (c)计算<br> {% raw %}$$\alpha_m = \log\frac{1-err_m}{err_m}$${% endraw %} (d)更新样本权重{% raw %}$w_i^{(m+1)} = w_i^{(m)} \cdot \exp \Big(\alpha_m I(y_i \neq b(x_i; \gamma_m))\Big),i=1,2,...,N${% endraw %}
> 3. 输出最终模型{% raw %}$f(x)=sign\Big( \sum_{m=1}^M \alpha_m b_m(x)\Big)${% endraw %}

我们惊奇地发现，这个算法和Adaboost一模一样。