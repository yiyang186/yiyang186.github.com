---
title: CenterNet笔记
date: 2019-05-20 07:33:00
categories:
  - 目标检测
tags: 
  - 深度学习
---

# Baseline & Motivation
CenterNet以CornerNet为baseline, CornerNet产生2个heatmaps(左上和右下Corner), 每个heatmap表达每个不同类别的keypoint的位置, 并赋予它们confience score, heatmap还生成embedding(用于辨别2个corner是否来自同一个物体), 和相对于每个corner的offset(offset学习输入图片到heatmap的映射). 为了生成物体的bounding boxes, 根据heatmaps的scores, 先挑选top-k个左上Corner和top-k个右下Corner, 再根据每对Corner的embedding向量的距离来确定每对Corner是否属于一个物体. 当embdding距离小鱼某阈值时, 生成一个bounding box, 并赋予它这对Corner的score的均值作为物体的score.
定义FD(false discovery = 1 - AP), 表示错误bounding box的比率, 如在coco上, FD[IoU=0.05]=32.7%, 意味着每100个bounding boxes有32.7个bounding boxes与ground-truth的IoU低于0.05. CornerNet的FD[IoU=0.5]=43.8. FD这么高的原因可能是CornetNet不能学习到 bounding boxes 的内部区域的视觉模式, two-stage detector里的RoI pooling可以帮助CornetNet学习bounding boxes内的视觉模式. 不过作者提出了更高效的one-stage 的 CenterNet, 对于每个物体, 使用3个keypoints表示, 继承了Rol pooling的功能, 且只关注物体中心区域的信息, 开销小, 关键在于center pooling和cascade corner pooling. 

# Object Detection as Keypoint Triplets
![CenterNet结构](https://wx1.sinaimg.cn/mw1024/3ff1a76fgy1g2zmisp903j20pd06bn0s.jpg)
CenterNet的结构, 在卷积网络主干上应用cascade corner pooling和center pooling来输出2个corner heatmaps和一个center keypoint heatmap. 与CornerNet相似, 一对corners和他们的embedding用于检测潜在的bounding box,然后center keypoints用于决定最终的bounding boxes. CenterNet在CornerNet的基础上, embed一个center keypoints的heatmap, 并预测center keypoint的offsets. 用类似CornerNet的方法生成top-k个bounding boxes. 为了过滤错误boungding boxes, (1)根据center keypoints的scores挑选top-k个center keypoints; (2)用相应的offsets重隐射center keypoints到输入图片; (3)为每个bounding boxes定义一个中心区域, 并检查这个中心区域是否包含相应类别的center keypoint; (4)若center keypoint在中心区域内, 保留这个bounding box, score为三个keypoint的平均score.
bounding box中心区域的大小影响检测结果. 例如, 对于小bounding boxes, 小中心区域导致低recall rate; 对于大bounding boxes, 大中心区域导致低precision. 因此它提出一个能使用bounding boxes大小的中心区域计算方法(scale-aware)(其实也就2种scale......)
![中心区域计算方法](https://wx3.sinaimg.cn/mw1024/3ff1a76fgy1g2zmiywwrgj20c206saal.jpg)
{% raw %}
$$ctl_x=\frac{(n+1)tl_x+(n-1)br_x}{2n}$$
$$ctl_y=\frac{(n+1)tl_y+(n-1)br_y}{2n}$$
$$cbr_x=\frac{(n-1)tl_x+(n+1)br_x}{2n}$$
$$cbr_y=\frac{(n-1)tl_y+(n+1)br_y}{2n}$$
{% endraw %}
上式tl=top-left, br=bottom-right, c=center, n是一个决定中心区域尺度的奇数, 文中当中心区域scale<150时, n=3; 当中心区域scale>=150时, n=5. 

# Enriching Center and Corner Information
## Center pooling
物体的几何中心不总能传达可识别的视觉模式(如人类的头包含更强的视觉模式, 而人的几何中心在身上), 所以他们提出center pooling捕捉更丰富,更可识别的视觉模式. 
![centerPooling](https://wx2.sinaimg.cn/mw1024/3ff1a76fgy1g2zmj2cx13j20p506wtdr.jpg)
(a)center pooling取横向和纵向的最大值;(b)conner pooling只取边界方向的最大值;(c)Cascade corner pooling取边界和内部方向的最大值.
对于不同边界的横向和纵向指的是:
topmost: 纵向向下
leftmost: 横向向右
bottommost: 纵向向上
rightmost: 横向向左

center pooling的原则是: 网络主干输出feature map, 为了确定feature map上的一个pixel是够是center keypoint, 需要找到该pixel所在横向和纵向方向上的最大值, 并加到一起.

## Cascade corner pooling
corner常常在物体外部, 它缺乏局部外形特征. CornetNet用corner pooling来处理这个问题, 其目标是找到边界方向上的最大值来确定corners. 但是这样会使conner对边缘敏感. 为了处理这个问题, 需要使corners看到物体的视觉模式, 于是提出cascade corner pooling, 它的原则是先沿着一个边界方向找最大值, 再以那个最大值为起点向目标内部方向找最大值, 最后把这两个最大值加到一起(另一个边界方向又有2个最大值). 
![cascadeCornerPooling](https://wx4.sinaimg.cn/mw1024/3ff1a76fgy1g2zmj5fz37j20hb06f3z3.jpg)
center pooling和cascade corner pooling的结构, 通过组合不同方向的corner pooling来实现. 如(a)横向的center pooling, 只需要串联left corner pooling和right cornet pooling; (b)为cascade top corner pooling, 它与CornerNet的不同时, 在top corner pooling前有一个left corner pooling.

# Training & Inference
## Training
- input image: 511 x 511
- headmaps size: 128 x 128
- optimization: Adam
- GPU: Tesla V100(32GB) x 8
- batch size: 48
- iteration: 480k
- learning rate: {% raw %}$2.5x10^(-4)${% endraw %} for 450k iter, {% raw %}2.5x10^(-5){% endraw %} for 30k iter
- training loss:
{% raw %}$$L = L_{co/det} + L_{ce/det} + αL_{co/pull} +βL_{co/push} + γ(L_{co/off} + L_{ce/off})$${% endraw %}
L=loss, co=corner, ce=center, det=detection, off=offset
L(co/det)和 L(ce/det) 是检测corner和center的focal loss, 细节参考ConerNet
L(co/pull)用于最小化同一物体的embedding vectors的距离, L(co/push)用于最大化不同物体的embedding vectors的距离, 细节参考ConerNet
L(co/off), L(ce/off) 分别是corner和center offset的L1 loss
系数α, β, γ取0.1, 0.1, 1

## Inference
- input: 原图和水平翻转后的图片
- single-scale: 1
- multi-scale: 0.6, 1, 1.2, 1.5, 1.8
- top-k 选keypoints: top-70 for center keypoints和2个corner
- bounding boxes: 水平翻转后的图片得到的bounding boxes还要水平翻转回来,再和原图的boungding boxes混合.
- nms: Soft-nms
- output: 根据score输出top-100 bounding boxes

# Comparisons with State-of-the-art Detectors
![evaluation](https://wx2.sinaimg.cn/mw1024/3ff1a76fgy1g2zmj87hpkj20qs0gqdki.jpg)

论文链接: [CenterNet: Object Detection with Keypoint Triplets](http://cn.arxiv.org/abs/1904.08189)
