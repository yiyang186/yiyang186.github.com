---
title: pthread中如何计时
date: 2016-04-06 15:01:00
categories:
  - 操作
tags: 
  - pthread
---

使用pthread编写的多线程程序，若是用clock_t结构体和clock()函数计时，多线程程序的运行时间会偏大，如
```
#include <time.h>
clock_t begin, end;
begin = clock();
// pthread多线程代码
end = clock();
printf("%f秒\n",  (double)(t1 - t0) / CLOCKS_PER_SEC);
```
这是因为clock()记录了所有CPU的时钟滴答数[^footer1], 求出的自然是所有Cores的使用时间之和。要计算实际并行计算的真实时间，应该使用lrt，如：
```
#include <time.h>
struct timespec begin, end;
double timediff;
clock_gettime(CLOCK_MONOTONIC, &begin);
// pthread多线程代码
clock_gettime(CLOCK_MONOTONIC, &end);
timediff = end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) / 1000000000.0;
printf("排序时间为%f秒\n",  timediff);
```
其中的timespec的结构如下
```
struct timespec {
    time_t tv_sec;     // 秒    
    long int tv_nsec;  // 纳秒   
};
```
编译时需加入参数-lrt, 如：
```
[root@slave02 yy]# g++ test.cpp -lrt -o test -lpthread
```
[^footer1]: 时钟滴答数（clock tick），从进程启动开始计时，因此这是相对时间。每秒钟包含CLOCKS_PER_SEC（time.h中定义的常量，一般为1000）个时钟滴答。时钟滴答数用数据类型clock_t表示。clock_t类型一般是32位整数类型。