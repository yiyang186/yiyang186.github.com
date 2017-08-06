---
title: 为Jupyter配置密码
date: 2016-03-16 16:51:00
categories:
  - 操作
tags: 
  - jupyter
---

在Python中生成密码

```
from notebook.auth import passwd
passwd('123456')
'sha1:1996ed5b2fc6:40da178c53092195aab3e1ce840e8c5c9e335fab'
```


修改jupyter配置文件

```
c = get_config() 
c.NotebookApp.ip = ‘*’ 
c.NotebookApp.password = u’sha1:1996ed5b2fc6:40da178c53092195aab3e1ce840e8c5c9e335fab’ 
c.NotebookApp.open_browser = False 
c.NotebookApp.port = 9999
```