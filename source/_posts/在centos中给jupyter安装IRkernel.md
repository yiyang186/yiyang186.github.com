---
title: 在centos中给jupyter安装IRkernel
date: 2017-12-28 09:10:00
categories:
  - 操作
tags: 
  - jupyter
  - R
---

# 环境：
- os: centos7
- R: 3.4.2
- jupyter: 4.4.0

# 坑预告
- R中无法安装git2r, devtools
- IRkernel::installspec()找不到jupyter

# 安装准备
确保下面这些工具已经被安装在系统中
```
yum install gcc gcc-c++ gcc-gfortran pcre-devel tcl-devel zlib-devel bzip2-devel readline-devel libXt-devel tk-devel tetex-latex gnutls-devel.x86_64 libcurl libcurl-devel libxml2 libxml2-devel openssl openssl-devel
```
# 安装步骤

1.**一定要sudo进入R**
```
sudo R
```

2.先试试git2r能不能安装成功
```
install.packages("git2r")
```
不过不行的话，退出R, 在外边下载安装git2r
```
git clone https://github.com/ropensci/git2r
sudu R CMD INSTALL git2r
```

3.在sudo回到R下，按照[官网文档](https://irkernel.github.io/installation/)的步骤做。
```
install.packages(c('repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))
devtools::install_github('IRkernel/IRkernel')
IRkernel::installspec(user = FALSE)
```

如果提示"jupyter-client has to be installed but “jupyter kernelspec --version” exited with code 127."。那么需要把已经安装好的jupyter链接到/usr/bin下，使R可以比较方便地找到jupyter
```
sudo ln -s /home/pyy/tool/anaconda3/bin/jupyter /usr/bin/jupyter
```
"/home/pyy/tool/anaconda3/bin/jupyter"是我的jupyter的完整路径

# 结果
最后，打开jupyter-lab里就自动出现R kernel了

![Markdown](http://i4.fuimg.com/602416/ad3f949165a86dcc.png)