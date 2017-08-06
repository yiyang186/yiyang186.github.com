---
title: centos安装caret包失败
date: 2016-08-28 10:02:00
categories:
  - 操作
tags: 
  - R
---

# 问题
实验室的redhat用的是centos源，跟centos情况应该差不多。

当我安装caret时，R提示我以下这些包安装失败
 ‘car’‘nloptr’, ‘lme4’, ‘pbkrtest’‘minqa’

理清依赖关系，最根源的问题是minqa和nloptr安装失败。

# 安装minqa
安装minqa失败的原因是ld找不到libgfortran.so, 由于我已经安装了gcc, 应该有libgfortran.so的。后来我在/usr/lib64/下边找到了libgfortran.so.3.0.0（你也可能会在/usr/lib或者/usr/local/lib、/usr/local/lib64里找到）
```
ln -sv /usr/lib64/libgfortran.so.3.0.0 /usr/lib64/libgfortran.so
```
然后再到R里安装minqa就行了。

# 安装nloptr
在stack overflow里搜到的办法是, 系统需要安装nlopt和nlopt-devel
```
yum install nlopt nlopt-devel
```
回到R就可以安装nloptr了

# 安装caret包
接下来安装caret包就很顺利了，要是不顺利的话，再把上述依赖包一个一个安装，看看是什么问题，一个一个解决。