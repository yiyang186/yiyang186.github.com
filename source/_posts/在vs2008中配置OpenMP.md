---
title: 在vs2008中配置OpenMP
date: 2016-03-23 11:37:00
categories:
  - 操作
tags: 
  - openmp
---

# 在vs2008上编译OpenMP
1. 试了好多方法，感觉直接在vs2008上写最简单，右键点击解决方案下面的项目名，在弹出的右键菜单选择最后一个“属性”。
![这里写图片描述](http://img.blog.csdn.net/20160323112356122)
2. 依次点击 配置属性 =》C/C++ =》语言，把右侧的OpenMP支持一栏的否改为是。这样你的OpenMP程序就可以编译了。
![这里写图片描述](http://img.blog.csdn.net/20160323112557920)
3. 如果运行程序时报错，找不到vcomp90.dll或者vcomp90d.dll，那么就到c:\windows\winsxs\目录下搜索vcomp90，并把vcomp90.dll和vcomp90d.dll这两个文件复制到你的项目根目录处，即有.vcproj的目录，而不是最外层的目录。这两个文件也可以到我的网盘下载
[点击进入网盘下载](http://pan.baidu.com/s/1skpWeKH)
![这里写图片描述](http://img.blog.csdn.net/20160323113217567)