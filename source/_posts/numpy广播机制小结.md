---
title: Numpy广播机制小结
date: 2017-11-09 21:09:23
categories:
  - 操作
tags: 
  - numpy
---

# 广播
numpy中数组的广播机制可以参考[Array Broadcasting in numpy](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)。numpy数组间的基础运算是element-by-element的，如a+b的结果就是a与b数组对应位相加，必须满足a.shape == b.shape。这要求维数相同，且各维度的长度相同。当运算中的2个数组的shape不同时，numpy将自动触发广播机制。如：
```python
>>> from numpy import array
>>> a = array([[ 0, 0, 0],
...            [10,10,10],
...            [20,20,20],
...            [30,30,30]])
>>> b = array([1,2,3])
>>> a + b
array([[  1,   2,   3],
       [ 11,  12,  13],
       [ 21,  22,  23],
       [ 31,  32,  33]])
```
![](http://scipy.github.io/old-wiki/pages/image0020619.gif?action=AttachFile&do=get&target=image002.gif)

4x3的二维数组与长为3的一维数组相加，等效于把数组b在二维上重复4次再运算
```python
>>> bb = np.tile(b, (4, 1))
>>> bb
array([[ 1,  2,  3],
       [ 1,  2,  3],
       [ 1,  2,  3],
       [ 1,  2,  3]])
>>> a + bb
array([[  1,   2,   3],
       [ 11,  12,  13],
       [ 21,  22,  23],
       [ 31,  32,  33]])
```

广播的概念可以理解为下面的C代码，b被广播到了更深层的for循环中，b中的每个元素都被重复使用了4次。
```C
int a[4][3] = {...};
int b[3] = {...};
int result[4][3] = 0;
for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 3; j++) {
        result[i][j] = a[i][j] + b[j];
    } 
}
```
但是，并不是任何维数不等的数组运算时都能触发广播机制，只有当两个数组的**trailing dimensions**（尾部维度）**compatible**（兼容）时才会触发广播，否则报错`ValueError: frames are not aligned exception`。<br>
什么是尾部维度？维数较大的数组的比维数较小的数组多出来的维度看做“头部维度”，剩下的就是“尾部维度”。将两数组的shape右对齐，右半边可以上下对应的部分就是尾部，如下面1,2轴。b数组将会在0轴广播256遍。
```
                   axis:   0     1   2
      a      (3d array): 256 x 256 x 3
      b      (2d array):       256 x 3
      a + b  (2d array): 256 x 256 x 3
```
什么是兼容？来自官方文档[Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)的解释很简练:
> Two dimensions are compatible when
> 1. they are equal, or
> 2. one of them is 1

即对应的尾部维度的长是相等的或者其中有一方是1。上面a,b对应的尾部维度是相等的，因此a,b的维度是兼容的，对应第一种情形。下面的a,b对应第二种情形：
```
                   axis:  0   1   2   3
      A      (4d array):  8 x 1 x 6 x 1
      B      (3d array):      7 x 1 x 5
      A + B  (2d array):  8 x 7 x 6 x 5
```
即“1”是一张万能牌，它与任意长度的维度兼容，A在1轴广播7遍，在3轴广播5遍，而B在0轴广播8遍。如果我们把0轴空着的部分看做1，那么一切都将使用第二冲情形"one of them is 1"。

> **Tips** <br>
> 对于任意shape的数组，我们可以利用np.newaxis使它们兼容，如下面的情况不会触发广播机制
> ```
>       a      (2d array):  3 x 4
>       b      (2d array):  5 x 4
>       a + b  ValueError
> ```
> 若转化为下面的形式（当然意义也变了，怎么转化应当根据需求而定），可以触发广播。
> ```
>       a[:, np.newaxis, :]      (3d array):  3 x 1 x 4
>       b                        (2d array):      5 x 4
>       a[:, np.newaxis, :] + b  (3d array):  3 x 5 x 4
> ```
> 

# 问题
广播机制可以帮助我们写出简短，直观的代码，借C的执行效率，比python循环有更多优势。但是它有一个严重的问题，它特别容易引起内存溢出。看下面这个问题<br>
在[c231n assignment1](http://cs231n.github.io/assignments2017/assignment1/)的knn部分中，需要计算测试集(500x3072)所有样本到训练集(5000x3072)所有样本的距离。题目要求不可使用python的循环，且不能用scipy中的函数，只能用numpy来算,结果是要得到距离矩阵dists(500x5000),其中dists[i, j]表示第i个测试样例与第j个训练样例的距离。
```python
def compute_distances_no_loops(self, X):
  """
  Compute the distance between each test point in X and each training point
  in self.X_train using no explicit loops.

  Input / Output: Same as compute_distances_two_loops
  """
  num_test = X.shape[0]
  num_train = self.X_train.shape[0]
  dists = np.zeros((num_test, num_train)) 
  #########################################################################
  # TODO:                                                                 #
  # Compute the l2 distance between all test points and all training      #
  # points without using any explicit loops, and store the result in      #
  # dists.                                                                #
  #                                                                       #
  # You should implement this function using only basic array operations; #
  # in particular you should not use functions from scipy.                #
  #                                                                       #
  # HINT: Try to formulate the l2 distance using matrix multiplication    #
  #       and two broadcast sums.                                         #
  #########################################################################
  pass
  #########################################################################
  #                         END OF YOUR CODE                              #
  #########################################################################
  return dists
```
如果我们直接使用广播机制
```python
def compute_distances_no_loops(self, X):
  dists = np.sqrt(((X[:, np.newaxis, :] - self.X_train) ** 2).sum(axis=-1))
  return dists
```
将会引发`Memory Error`。X[:, np.newaxis, :] - self.X_train的shape是500 x 5000 x 3072，如果数组中每个元素为4字节的整数，那么它将占用 500 x 5000 x 3072 x 4 / (2 ^ 30) = 28.6GB的空间，严重超过一般的内存空间(我的电脑是16GB)。就算不超过虚拟内存空间，超过物理内存空间也会由于频繁的换页而严重拖慢执行效率，比python的for循环还要低效！
```
               X[:, np.newaxis, :]   (3d array): 500 x    1 x 3072
                      self.X_train   (2d array):       5000 x 3072
X[:, np.newaxis, :] - self.X_train   (3d array): 500 x 5000 x 3072
```

解决的办法是将平方拆开：$(X-Xtrain)^2 = X^2 + Xtrain^2-2X \cdot Xtrain$:
```python
def compute_distances_no_loops(self, X):
  dists = ((X ** 2).sum(axis=-1)[:, np.newaxis]  
          + (self.X_train ** 2).sum(axis=-1) 
          - 2 * X.dot(self.X_train.T))
  return dists
```
这里还用了个小技巧：
```
                         X[:, np.newaxis, :]: 500 x    1 x 3072
                                     X_train:       5000 x 3072
               X[:, np.newaxis, :] * X_train: 500 x 5000 x 3072
(X[:, np.newaxis, :] * X_train).sum(axis=-1):        500 x 5000
```
等价于
```
                       X   (2d array):  500 x 3072
                 X_train   (2d array): 5000 x 3072
        X.dot(X_train.T)   (2d array):  500 x 5000
```
举(zai)个(xia)例(mei)子(neng)会(li)清(biao)楚(shu)一(qing)些(chu):
```python
>>> a = np.random.randint(9, size=(2, 4))
>>> a
array([[5, 8, 6, 7],
       [7, 3, 0, 0]])
>>> b = np.random.randint(9, size=(2, 4))
>>> b
array([[4, 8, 5, 8],
       [5, 5, 5, 5]])
>>> a[:, np.newaxis, :] * b
array([[[20, 64, 30, 56],
        [25, 40, 30, 35]],

       [[28, 24,  0,  0],
        [35, 15,  0,  0]]])
>>> (a[:, np.newaxis, :] * b).sum(axis=-1)
array([[170, 130],
       [ 52,  50]])   
>>> a.dot(b.T)
array([[170, 130],
       [ 52,  50]])
```

# 总结
1. 只有当两个数组的尾部维度兼容时才会触发广播；
2. 两个维度的长度是相等的或者有一方长度为1则称这两个维度兼容；
3. 广播机制可以提高代码执行效率并简化代码，但当广播后得到的中间结果所占空间大于空闲物理内存时，效率变得非常低，当中间结果所需空间大于可用虚拟内存时，内存溢出，引发`Memory Error`。

<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: 'numpy_broadcasting',
  title: 'Numpy广播机制小结',
  owner: 'yiyang186',
  repo: 'blog_comment',
  oauth: {
    client_id: '2786ddc8538588bfc0c8',
    client_secret: '83713f049f4b7296d27fe579a30cdfe9e2e45215',
  },
})
gitment.render('container')
</script>