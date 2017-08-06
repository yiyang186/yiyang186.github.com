---
title: 在mac终端上执行R脚本
date: 2016-03-06 10:31:00
categories:
  - 操作
tags: 
  - R
---

1. 创建file.R文件
2. 键入R代码
3. 在输出图像（而非保存文本）如png(),jpeg(),...等函数后，一定要加上一行
	```
	dev.off()
	```
	关闭虚拟显示设备。
	
4. 注意将数据处理结果输出到文件
	```
	write.table(x=df , file=output.csv , sep=',', row.names=FALSE, quote=FALSE)
	```
	
5. 保存成.R文件
6. 在工作目录下，在终端中输入以下命令:(xxx是脚本文件名，其他的照抄)
	```
	R CMD BATCH --args xxx.R
	```

7. 等待执行结束。用这种方式执行脚本，终端不会输出执行过程代码，这些代码会保存到xxx.Rout文件中