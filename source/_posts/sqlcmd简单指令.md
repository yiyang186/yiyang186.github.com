---
title: sqlcmd简单指令
date: 2016-05-27 13:51:00
categories:
  - 操作
tags: 
  - sql
---

# 前提

- sql server 2008
- 注意1433端口是否打开[^1]

# 登陆数据库
sqlcmd -S 数据库服务器地址 -U 用户名 -P 密码
或
sqlcmd -S 数据库服务器地址\数据库实例名 -U 用户名 -P 密码

# 查询数据库
1. 查看有哪些数据库
\>1 SELECT name FROM sys.databases
\>2 GO
每次输入sql语句都要打一行go, 下面不再赘述，sys.databases是系统默认的全局变量（学过sql, 但没专门学过sql server，姑且这么叫吧）

2.  切换数据库上下文
USE 数据库名

3.  查看数据库里有哪些表
SELECT name FROM sysobjects

4.  查看表中的列名
SELECT name FROM syscolumns WHERE id=object_id('表名')

5. 其他
sql server用的sql语句叫做Transact-SQL（T-SQL）, 跟sql还有点不一样，具体请参见[T-SQL经典语句](http://imdbt.blog.51cto.com/903896/218590)，以及[官方文档](https://msdn.microsoft.com/zh-cn/library/ms177563(v=sql.100).aspx)。

# 从CSV文件导入数据库
先为要导入的数据建表
```sql
CREATE TABLE 表名( 
	列1 NVARCHAR(MAX), 
	列2 NVARCHAR(MAX), 
	列3 NVARCHAR(MAX) 
) 
```

从csv文件中导入数据
```sql
BULK INSERT 表名
FROM 'csv文件路径名'
WITH(
	FIELDTERMINATOR = ',',
	ROWTERMINATOR = '\n'
)
```

[^1]: 到sql server configuration manager里的network configuration里去设置tcp/ip的端口（IP Address选项卡拉到最下面）为1433，再重启sql server services即可，具体步骤请参见[解决sqlserver 2008 sqlcmd无法登陆](http://www.cnblogs.com/skynothing/archive/2010/08/26/1809125.html)
