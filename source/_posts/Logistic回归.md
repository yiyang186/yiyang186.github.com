---
title: 逻辑回归
date: 2017-05-01 19:53:00
categories:
  - 机器学习
tags: 
  - ESL
  - 逻辑回归
---

# 逻辑回归的建模
## sigmoid函数
说道建模，总有一些粗暴的地方，即强行给真实模型{% raw %}$f${% endraw %}指定某种形式{% raw %}$\hat f${% endraw %}，像线性回归里就强行指定了{% raw %}$\hat f(x)=\beta^T x, x \in R^p, \beta \in  R^{p+1}${% endraw %}这种形式，然后依靠所找到的最优的{% raw %}$\beta${% endraw %}使{% raw %}$\hat f${% endraw %}尽量接近{% raw %}$f${% endraw %}。
分类问题与回归的问题的不同在于y的值域不同，对于分类问题，我们想把样本x分到{% raw %}$y\in \{a,b,c,d,...\}${% endraw %}的类别中。这样，我们希望拟合y在给定x下的条件概率{% raw %}$P(y=a,b,...|x)${% endraw %}。概率的值域为[0,1]。由于我们不能保证内积的结果在[0,1]之内，所以线性回归的那种内积的形式就不适用了，我们就不能以这种形式的模型去拟合真实模型。
人们找到一种叫sigmoid(s型)的函数
![这里写图片描述](http://img.blog.csdn.net/20170621150508657?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcTF3MmUzcjQ0NzA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
{% raw %}$$y=g(x)=\frac{1}{1+e^{-x}}, x\in R, y\in(0,1)\tag 1$${% endraw %}
它可以把范围在{% raw %}$(-\infty, +\infty)${% endraw %}的值映射到{% raw %}$(0,1)${% endraw %}里去，由于它是单调递增函数，因此可以保持输入值的单调性、奇偶性、周期型。把{% raw %}$\hat f(x)=\beta^T x${% endraw %}带入其中得到
{% raw %}$$h_\beta(x)=g(\beta^Tx)=\frac{1}{1+e^{-\beta^T x}}\tag 2$${% endraw %}

## 二分类
对于二分类问题{% raw %}$y\in \{0,1\}${% endraw %}，我们**假设**
{% raw %}$$
\begin{equation}
\left\{
\begin{array}{c}
P(y=1|x; \beta) &=& h_\beta(x)=\frac{1}{1+e^{-\beta^T x}}\\
P(y=0|x; \beta) &=& 1 - h_\beta(x)=\frac{e^{-\beta^T x}}{1+e^{-\beta^T x}}\\
\end{array}
\right.
\end{equation}
x \in R^p, \beta \in  R^{p+1}
\tag 3$${% endraw %}
可能因为这种模型的结果是在0, 1之间，且保持了线性回归模型的内积形式，因此它被称作logistic regression，即逻辑回归，或逻辑斯蒂回归。


## 逻辑回归的特质
将(3)中的2个等式取对数相比（log-odd）得
{% raw %}$$\log{{P(y=1|x)}\over {P(y=0|x)}}=\beta ^Tx \tag4$${% endraw %}
可以发现以下特质，**考虑任意样本{% raw %}$x_0${% endraw %}**：

1. 当{% raw %}$P(y=1|x_0)>P(y=0|x_0)${% endraw %}时，{% raw %}$x_0${% endraw %}被分类到{% raw %}$\beta^Tx=0${% endraw %}的>0的一边;
2. 当{% raw %}$P(y=1|x_0)$ < $P(y=0|x_0)${% endraw %}时，{% raw %}$x_0${% endraw %}被分类到{% raw %}$\beta^Tx=0${% endraw %}的<0的一边;
3. 当{% raw %}$P(y=1|x_0)=P(y=0|x_0)${% endraw %}时，{% raw %}$x_0${% endraw %}在平面{% raw %}$\beta^Tx=0${% endraw %}上.

可见Logistic Regression仍然是一种线性方法，即用平面({% raw %}$\beta ^Tx=0${% endraw %})来分类。在二分类问题中，样本在超平面的一边就属于一类, 这个平面就是类与类之间的边界。
可以认为逻辑回归实际上式以(4)建模的。它认为一个样本属于类1的概率P(y=1|x)大的话，它应当在分类平面{% raw %}$\beta^Tx=0${% endraw %}的一边，即满足{% raw %}$\beta^Tx>0${% endraw %}；当这个样本属于0类概率 P(y=0|x)大时，它应当在分类平面{% raw %}$\beta^Tx=0${% endraw %}的另一边，即满足{% raw %}$\beta^Tx<0${% endraw %} 。并且，它严格地假设了两类条件概率的对数比是线性的，可以被输入X线性表出。我们知道由于类密度高斯假设和公共协方差假设，LDA(线性判别分析)的log-odd是x的线性函数，与(4)形式一样。**而Logistic绕过了这2个假设以式(4)建模**。逻辑回归不要求样本满足类密度高斯假设和公共协方差假，更具一般性。

其实，我们忘掉sigmoid函数，直接拿(4)来建模，粗暴地假设类对数比率(log-odd)是关于x的线性函数，又由于样本属于两类的概率之和为1
{% raw %}$$P(y=1|x)+P(y=0|x)=1\tag5$${% endraw %}

联立(4)(5)，解未知数为P(Y=1|x)和P(Y=-1|x)的方程还是可以得到式(3)，自然含有sigmoid函数。

## 多分类
对于K分类问题，如{% raw %}$y\in \{1,2,...,K\}${% endraw %}，我们希望模型拟合的是y的后验分布{% raw %}$P(y=i|x), i=1,2,...,K${% endraw %}。像二分类一样，如法炮制，假设两两类的条件概率对数比(log-odd)是关于x的线性函数，这样我们能列{% raw %}$\binom{K}{2}=\frac{K(K-1)}{2}${% endraw %}个方程。可是我们只有K个未知数，只需要列K个方程就行了，不妨取1,2,...,K-1类分别与K类的对数比来列方程，并限制样本属于K个类的概率的和为1，则
{% raw %}$$
\begin{equation}
   \left\{
   \begin{array}{c}
   \log{{P(y=1|x)}\over {P(y=K|x)}} &=&\beta_1 ^Tx\\
   \log{{P(y=2|x)}\over {P(y=K|x)}} &=&\beta_2 ^Tx\\
   ...\\
   \log{{P(y=K-1|x)}\over {P(y=K|x)}} &=&\beta_{K-1} ^Tx\\
   \sum_{i=1}^KP(y=i|x) &=& 1
   \end{array}
  \right.
\end{equation}
x \in R^p, \beta_k \in  R^{p+1}, k=1,2,...,K-1
\tag 6$${% endraw %}
解方程组得到多分类的模型
{% raw %}$$
\begin{equation}
  \left\{
   \begin{array}{c}
   P(y=k|x; \beta) &=& \frac{e^{-\beta_k^T x}}{1+\sum_{i=1}^{K-1}e^{-\beta_i^T x}}\\
   P(y=K|x; \beta) &=& \frac{1}{1+\sum_{i=1}^{K-1}e^{-\beta_i^T x}}\\
   \end{array}
  \right.
\end{equation}
\tag 7$${% endraw %}
其中，{% raw %}$x \in R^p, \beta_k \in  R^{p+1}, k=1,2,...,K-1${% endraw %}。
这很像**softmax回归**，把(7)中的“1”替换成“{% raw %}$e^{-\beta_K^T x}${% endraw %}”就成了softmax回归。
{% raw %}$$P(y=k|x; \beta) = \frac{e^{-\beta_k^T x}}{\sum_{i=1}^Ke^{-\beta_i^T x}}, k=1,2,...,K \tag 8$${% endraw %}
softmax回归可以看做是logistic的推广。当然softmax回归并不是这么推倒出来的，它的建模过程有它自己的考虑，像logistic一样出于某种需求被逐步构建出，就像你做deep learning搭积木一样，是一种建模过程。你能根据模型的某种数学特点把他们归类在一起，比如GLM, tree。模型与模型间确定的推导关系是很难给出的, 至少我在写这个博客时还做不到。或者我该以另一种形式开头，如“人们找到一种叫做softmax的函数，它有这样那样的特点。。。”再写一篇博客。

# 逻辑回归的求解
用最大似然估计法估计{% raw %}$\beta${% endraw %}, 令
{% raw %}$$p_i=P(y_i=1|x_i)=1-sigmoid(\beta^Tx_i)$${% endraw %}
则
{% raw %}$$P(y_i=0|x_i)=1-p_i=sigmoid(\beta^Tx_i)$${% endraw %}
那么似然函数为：
{% raw %}$$l(\beta)=\prod_{i=1}^np_i^{y_i}(1-p_i)^{1-y_i}$${% endraw %}
对数似然为：
{% raw %}$$L(\beta)=\log l(\beta)=\sum_{i=1}^ny_i\log p_i+(1-y_i)\log(1-p_i)\tag8$${% endraw %}
最大化{% raw %}$L(\beta)${% endraw %}即最小化{% raw %}$-L(\beta)${% endraw %}
{% raw %}$$-L(\beta)=-\log l(\beta)=\sum_{i=1}^n-y_i\log p_i-(1-y_i)\log(1-p_i)\tag9$${% endraw %}
(9)等号右边每一项为交叉熵（cross entropy）,因此对逻辑回归使用最大似然估计等价于最小化交叉熵，因此在神经网络中以交叉熵为损失函数求解二分类问题与最大似然估计是等价的。
这里只介绍一般逻辑回归的求解，回到(8), 将(6)(7)带入(8)得
{% raw %}$$L(\beta)=\sum_{i=1}^ny_i\beta^Tx_i-\log(1+exp(\beta^Tx_i))$${% endraw %}
令其梯度为0有
{% raw %}$$\frac{\partial L(\beta)}{\partial \beta}=\sum_{i=1}^nx_i(y_i-\frac{exp(\beta^Tx_i)}{1+exp(\beta^Tx_i)})=\sum_{i=1}^nx_i(y_i-p_i)=0\tag{10}$${% endraw %}
(10)是非线性方程组，难以求得{% raw %}$\beta${% endraw %}的解析解，可使用Newton-Raphson算法（牛顿迭代法）求解(10)的零点，Newton-Raphson算法需要(10)的导数，也就是{% raw %}$L(\beta)${% endraw %}的Hessian矩阵（二阶导数）
{% raw %}$$\frac{\partial L^2(\beta)}{\partial \beta \partial \beta^T}=-\sum_{i=1}^nx_ix_i^Tp_i(1-p_i)$${% endraw %}
因此Newton-Raphson算法每一步的迭代公式为：
{% raw %}$$\beta^{new}=\beta^{old}-\bigg(\frac{\partial L^2(\beta)}{\partial \beta \partial \beta^T}\Bigg|_{\beta^{old}}\bigg)^{-1}\frac{\partial L(\beta)}{\partial \beta}\Bigg|_{\beta^{old}}\tag{11}$${% endraw %}
可根据(11)迭代即可求解出(10)的零点{% raw %}$\beta${% endraw %}，同时也是(8)的极值点。

# 最小化交叉熵损失与最大化似然函数
最大化似然估计是显而易见的：
{% raw %}$$\max_{p_i} \sum_{i=1}^n y_i\log p_i+(1-y_i)\log(1-p_i)$${% endraw %}

那么，**如何理解最小化交叉熵(互熵)损失**?<br>
假设给定x,y真实的分布服从0-1分布，假设y取1的真实条件分布为p(y|x)有，$y|x \sim p(y|x)$。我们所估计的条件分布为$\hat p(y|x)$。如果这两个分布很接近的话，他们的KL散度应该尽量小，那么可以最小化KL散度为目标，找到最接近$p(y|x)$的$\hat p(y|x)$：

{% raw %}
$$\begin{aligned}\min_{\hat p} D_{KL}(p \|\hat p)
  &= \min_{\hat p} E_p \log \frac{p(y|x)}{\hat p(y|x)} \\
  &= \min_{\hat p} \{E_p \log p(y|x) - E_p\log \hat p(y|x)\} \\
  &= \min_{\hat p} \{E_p \log p(y|x)\} + \min_{\hat p} \{- E_p\log \hat p(y|x)\} \\
  &= \min_{\hat p} \{-E_p\log \hat p(y|x)\} \\
\end{aligned} \tag{12}$$
{% endraw %}

由于{% raw %}$E_p \log p(y|x)${% endraw %}与$\hat p$无关，所以式(12)的最后一个等号成立。其实根据交叉熵的定义，这是显而易见的:
{% raw %}$$H(p,\hat p)=H(p)+D_{KL}(p \| \hat p) \tag{13}$${% endraw %}

其中, $H(p,\hat p)$是分布p与分布$\hat p$的交叉熵，H(p)是分布p的信息熵。由于数据集已知，H(p)已知，所以最小化KL散度{% raw %}$D_{KL}(p \| \hat p)${% endraw %}等价于最小化交叉熵$H(p,\hat p)$.<br>
把交叉熵展开，可得到最小化交叉熵与最大化似然函数等价：
{% raw %}
$$\begin{aligned}\min_{\hat p} \{-E_p\log \hat p(y|x)\}
  &= \min_{\hat p} \{-E_{y|x}\log \hat p(y|x)\} \\
  &= \min_{p_i} \frac{1}{n} \sum_{i=1}^n \bigg\{ \begin{matrix} -\log p_i & y_i=1|x_i \\ -\log (1-p_i) & y_i=0|x_i \end{matrix} \\
  &= \min_{p_i} \frac{1}{n} \sum_{i=1}^n -y_i\log p_i -(1-y_i)\log (1-p_i) \\
  &= \min_{p_i} \sum_{i=1}^n -y_i\log p_i -(1-y_i)\log (1-p_i) \\
  &= \max_{p_i} \sum_{i=1}^n y_i\log p_i +(1-y_i)\log (1-p_i) \\
\end{aligned} \tag{14}$$
{% endraw %}


# 以{% raw %}$\log{{P(y=1|x)}\over {P(y=0|x)}}=\beta^Tx${% endraw %}建模的遐想

**考虑任意样本{% raw %}$x_0${% endraw %}**：

- 若{% raw %}$x_0${% endraw %}属于1类的概率大于属于0类的概率，它应当在分类平面{% raw %}$\beta^Tx=0${% endraw %}的某一边，不妨设{% raw %}$\beta^Tx_0>0${% endraw %}；
- 若{% raw %}$x_0${% endraw %}属于0类的概率大于属于1类的概率，它应当在分类平面{% raw %}$\beta^Tx=0${% endraw %}的另一边，即{% raw %}$\beta^Tx_0 < 0${% endraw %}。

换成数学的语言就是：

如果|那么
------|-------------
{% raw %}$$\log{{P(y=1|x_0)}\over {P(y=0|x_0)}}>0$${% endraw %} | {% raw %}$$\beta^Tx_0>0$${% endraw %}
{% raw %}$$\log{{P(y=1|x_0)}\over {P(y=0|x_0)}} < 0$${% endraw %} | {% raw %}$$\beta^Tx_0 < 0$${% endraw %}

只要{% raw %}$\log{{P(y=1|x_0)}\over {P(y=0|x_0)}}${% endraw %}和{% raw %}$\beta^Tx_0${% endraw %}同号就是我们想要的模型。**Logistic直接以{% raw %}$\log{{P(y=1|x)}\over {P(y=0|x)}}=\beta^Tx${% endraw %}建模**保证了这种同号的要求，但是这样建模多了一个副产品，就是“绝对值相等”——
{% raw %}$$|\log{{P(y=1|x_0)}\over {P(y=0|x_0)}}|=|\beta^Tx_0|\tag{15}$${% endraw %}
容易理解，{% raw %}$|\beta^Tx_0|${% endraw %}是{% raw %}$x_0${% endraw %}到分类平面{% raw %}$\beta^Tx=0${% endraw %}的距离，而{% raw %}$|\log{{P(y=1|x_0)}\over {P(y=0|x_0)}}|${% endraw %}是样本{% raw %}$x_0${% endraw %}属于于0，1类概率的对数比率（log-odd）。二者相等吗？**样本到分类平面的距离与样本属于各类概率的对数比率大小相等**吗？有可能碰到正好满足的样本，但是绝大多数情况下不相等。这是逻辑回归建模稍稍强加于实际模型的假设。

这些想法由ESL和[我在知乎中关于逻辑回归的回答](https://www.zhihu.com/question/35322351/answer/141562541)引申而来，我也没见过相关的文献，要是有相关的文献作为以上臆想的佐证或者驳斥，烦请通知我。

# Newton-Raphson算法(牛顿迭代法)
![牛顿迭代法](https://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif)


<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: 'logistic_regression',
  title: 'logistic回归',
  owner: 'yiyang186',
  repo: 'blog_comment',
  oauth: {
    client_id: '2786ddc8538588bfc0c8',
    client_secret: '83713f049f4b7296d27fe579a30cdfe9e2e45215',
  },
})
gitment.render('container')
</script>
