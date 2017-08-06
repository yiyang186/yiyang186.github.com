---
title: python解压rar文件
date: 2016-07-08 11:49:00
categories:
  - 操作
tags: 
  - python
---

# 应用场景
在数据预处理阶段，有时候会发现我们的数据存储在大量杂乱无章的压缩文件中，这些压缩文件还可能处在复杂的目录树结构下。这时候你可能会想写个python脚本来处理。
对于zip文件，python 的zipfile模块提供了很好的支持，但是对于rar格式的压缩文件，要麻烦一点。


# 安装unrar
[unrar](https://pypi.python.org/pypi/unrar/)是python下支持解压rar文件的插件
```
pip install unrar
```
不过这个插件需要rarlib的支持，不然无法解压


# 安装rarlib
到[这个网页](http://www.rarlab.com/rar_add.htm)去下载相应版本的rarlib，由于我用的是windows系统，所以下载了UnRAR.dll。
我照着unrar文档中的方式设置UnRAR.dll的环境变量，结果毫无效果，看看报错，摸索出来了一个方法。
将安装好后的rarlib目录加入环境变量path（由于我的是64bit系统，所以只加入了rarlib目录下的64子目录）。由于unrar模块要搜索名为unrar.dll的文件，所以需要把目录中的UnRAR.dll， UnRAR.lib（UnRAR64.dll, UnRAR64.lib）改为unrar.dll，unrar.lib
这样unrar就可以顺利解压rar文件了