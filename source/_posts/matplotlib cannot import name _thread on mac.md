---
title: matplotlib cannot import name _thread on mac
date: 2016-03-01 19:55:00
categories:
  - 操作
tags: 
  - matplotlib
---

最后的2行错误信息是
```
    from six.moves import _thread
ImportError: cannot import name _thread
```
发现是six出现了问题，用pip更新一下six，问题并没有解决，原因是并没有真正更新six的文件。
在python下输入：
```
import six
print six.__file__
```
/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/six.pyc

这是我们的python实际使用的six，而我们手动更新的six却是装在/Library/Python/2.7/site-packages/，我们把six.\__file\__的文件删除掉，python就只能用我们更新的six了
```
sudo rm -rf /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/six.*
```
重启ipython/python就行了，如果之前并未有更新six, 应该在这一步中更新six。
```
sudo pip install --upgrade six
```

另外，在很多时候我们希望忽略过去下载的安装包，直接下载安装可以使用--ignore-installed这个参数，比如我发现的的matplotlib的mplot3d部分有点问题，我想再重新下载安装一遍，可以这么做
```
sudo pip install --upgrade --ignore-installed matplotlib
```
这会把相关的包（numpy, pytz, six, python-dateutil, cycler, pyparsing, matplotlib）都下载安装一遍