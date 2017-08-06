---
title: 怎样用scipy求一些统计量的p值和分位数
date: 2016-10-06 11:05:00
categories:
  - 操作
tags: 
  - scipy
---

scipy.stats模块中有不少涉及计算统计量的子模块
如
scipy.stats.uniform
scipy.stats.norm
scipy.stats.t
scipy.stats.chi2
scipy.stats.f
更多子模块参见[这里](http://docs.scipy.org/doc/scipy/reference/stats.html)

其中scipy.stats.f内有如下方法：
![图一](http://img.blog.csdn.net/20161006104131553)
来自于[这里](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html#scipy.stats.f)

这些方法在其他子模块中都大同小异。这里重点介绍求F统计量的p值和分位数，t统计量，卡方，正态，的求法基本类似。

例1：求{% raw %}$Pr(F_{4,58}>1.67)=?${% endraw %},即已知临界值求p值
```
>>> from scipy.stats import f
>>> f.sf(1.67, 4, 58)
0.16927935111708425
>>> 1 - f.cdf(1.67, 4, 58)
0.16927935111708448
```

例2：求{% raw %}$F_{4,58}^{(1-0.17)}=?${% endraw %} ,即已知p值求临界值
```
>>> from scipy.stats import f
>>> f.isf(0.17, 4, 58)
1.666945416681088
>>> f.ppf(1 - 0.17, 4, 58)
1.666945416681088
```

其中，isf是sf的逆运算， ppf是cdf的逆运算。具体解释参见上图。其他统计量的相关计算方法与之类似，不同之处就是少了自由度作为参数。