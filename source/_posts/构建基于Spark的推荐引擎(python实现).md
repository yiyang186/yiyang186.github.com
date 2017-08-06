---
title: 构建基于Spark的小推荐系统demo
date: 2015-10-18 21:28:00
categories:
  - 大数据
tags: 
  - spark
  - python
  - 推荐
---

这几天在看由人民邮电出版社出版的《Spark机器学习》（Machine Learning with Spark，Nick Pentreath），看的很是郁闷。这本书一会儿用python, 一会儿用scala。由于我很喜欢用python, 所以用Python把这本书的scala代码又实现了一遍。收获很大，比直接对着书敲scala要理解的更多，对整个推荐过程的理解更加深刻。当然，与实际应用中的推荐系统相比，这只是个玩具而已。

以下为我的python代码

```python
rawData = sc.textFile("hdfs:///user/yy/ml-100k/u.data")
rawData.first()
```
u'196\t242\t3\t881250949'
```python
# 取出数据，仅需要用户、电影、评分
rawRatings = rawData.map(lambda line: line.split('\t')[0:3])
rawRatings.first()
```
[u'196', u'242', u'3']
```python
import pyspark.mllib.recommendation as rd

# 由于ALS模型需要由Rating记录构成的RDD作为参数，因此这里用rd.Rating方法封装数据
ratings = rawRatings.map(lambda (user, movie, rating): rd.Rating(int(user), int(movie), float(rating)))
ratings.first()
```
Rating(user=196, product=242, rating=3.0)
```python
# 训练ALS模型
model = rd.ALS.train(ratings, 50, 10, 0.01)
model.userFeatures
```
<bound method MatrixFactorizationModel.userFeatures of <pyspark.mllib.recommendation.MatrixFactorizationModel object at 0x7f68d9b3fbd0>>
```python
# 利用该模型预测789用户对123电影的评分
predictedRating = model.predict(789, 123)
print predictedRating
```
3.00704909501



```python
# 给789用户推荐的前10个商品
topKRecs = model.recommendProducts(789, 10)
topKRecs
```
[Rating(user=789, product=708, rating=5.645492560638761),
     Rating(user=789, product=482, rating=5.632324542725622),
     Rating(user=789, product=502, rating=5.621572563993868),
     Rating(user=789, product=603, rating=5.5589276689720055),
     Rating(user=789, product=23, rating=5.552174994200555),
     Rating(user=789, product=182, rating=5.429093418196553),
     Rating(user=789, product=484, rating=5.424060744309293),
     Rating(user=789, product=479, rating=5.416303074919905),
     Rating(user=789, product=1020, rating=5.350687998038177),
     Rating(user=789, product=494, rating=5.341673625528914)]




```python
movies = sc.textFile('hdfs:///user/yy/ml-100k/u.item')
movies.first()
```
u'1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0'




```python
movies_fields = movies.map(lambda line: line.split('|'))
title_data = movies_fields.map(lambda fields: (int(fields[0]), fields[1])).collect()
titles = dict(title_data)
titles[123]
```
u'Frighteners, The (1996)'




```python
# keyBy从ratings RDD创建一个键值对RDD，选取user为主键
# lookup返回给定键的数据
moviesForUser = ratings.keyBy(lambda rating: rating.user).lookup(789) # 返回的是list
moviesForUser[0:10] # 结果为789对他看过的电影给出的评分
```
[Rating(user=789, product=1012, rating=4.0),
     Rating(user=789, product=127, rating=5.0),
     Rating(user=789, product=475, rating=5.0),
     Rating(user=789, product=93, rating=4.0),
     Rating(user=789, product=1161, rating=3.0),
     Rating(user=789, product=286, rating=1.0),
     Rating(user=789, product=293, rating=4.0),
     Rating(user=789, product=9, rating=5.0),
     Rating(user=789, product=50, rating=5.0),
     Rating(user=789, product=294, rating=3.0)]




```python
# sorted(list, key=lambda..., reverse=True)是对list的排序函数
# sc.parallelize(data)并行化数据，转换为rdd后才能用map方法
moviesForUser = sorted(moviesForUser, key=lambda r: r.rating, reverse=True)[0:10]
```


```python
[(titles[r.product], r.rating) for r in moviesForUser]
```
[(u'Godfather, The (1972)', 5.0),
     (u'Trainspotting (1996)', 5.0),
     (u'Dead Man Walking (1995)', 5.0),
     (u'Star Wars (1977)', 5.0),
     (u'Swingers (1996)', 5.0),
     (u'Leaving Las Vegas (1995)', 5.0),
     (u'Bound (1996)', 5.0),
     (u'Fargo (1996)', 5.0),
     (u'Last Supper, The (1995)', 5.0),
     (u'Private Parts (1997)', 4.0)]




```python
[(titles[r.product], r.rating) for r in topKRecs]
```
[(u'Sex, Lies, and Videotape (1989)', 5.645492560638761),
     (u'Some Like It Hot (1959)', 5.632324542725622),
     (u'Bananas (1971)', 5.621572563993868),
     (u'Rear Window (1954)', 5.5589276689720055),
     (u'Taxi Driver (1976)', 5.552174994200555),
     (u'GoodFellas (1990)', 5.429093418196553),
     (u'Maltese Falcon, The (1941)', 5.424060744309293),
     (u'Vertigo (1958)', 5.416303074919905),
     (u'Gaslight (1944)', 5.350687998038177),
     (u'His Girl Friday (1940)', 5.341673625528914)]




```python
itemId = 567
itemVec = model.productFeatures().lookup(itemId)[0]
```


```python
from scipy.spatial.distance import cosine

# cosine函数实际上是是求1-cosine,便于我们后面的排序，对1-cosine从小到大排等价于对cosine从大到小排
sims = model.productFeatures().map(lambda (id, factorVec): (id, cosine(factorVec, itemVec)))
sims.first()
```
(16, 0.48041293748773517)




```python
sortedSims = sims.sortBy(lambda s: s[1]).take(10)
sortedSims
```
[(567, 0.0),
     (413, 0.27774090532944073),
     (24, 0.28293701045346031),
     (184, 0.29302471333565439),
     (352, 0.29389332298954041),
     (1376, 0.30311195834153437),
     (201, 0.30636099292020724),
     (741, 0.31300444242427816),
     (685, 0.31398875037205187),
     (686, 0.31444575073270353)]




```python
print titles[itemId]
[(titles[id], sim) for (id, sim) in sortedSims]
```
Wes Craven's New Nightmare (1994)

[(u"Wes Craven's New Nightmare (1994)", 0.0),
     (u'Tales from the Crypt Presents: Bordello of Blood (1996)',
      0.27774090532944073),
     (u'Rumble in the Bronx (1995)', 0.28293701045346031),
     (u'Army of Darkness (1993)', 0.29302471333565439),
     (u'Spice World (1997)', 0.29389332298954041),
     (u'Meet Wally Sparks (1997)', 0.30311195834153437),
     (u'Evil Dead II (1987)', 0.30636099292020724),
     (u'Last Supper, The (1995)', 0.31300444242427816),
     (u'Executive Decision (1996)', 0.31398875037205187),
     (u'Perfect World, A (1993)', 0.31444575073270353)]




```python
actualRating = moviesForUser[0]
predictedRating = model.predict(789, actualRating.product)
squaredError = (actualRating.rating - predictedRating) ** 2
print "实际评分: %f, 预测评分: %f, 方差: %f. " % (actualRating.rating, predictedRating, squaredError)
```
实际评分: 5.000000, 预测评分: 5.042951, 方差: 0.001845. 



```python
usersProducts = ratings.map(lambda r: (r.user, r.product))
# predictAll方法以对(int, int)形式的rdd作为参数，这点与scala不同，scala直接用predict
predictions = model.predictAll(usersProducts).map(lambda r: ((r.user, r.product), r.rating))
predictions.first()
```
((368, 320), 4.883538186965982)




```python
# 形成一个(user, movie)做主键，(实际评分，预测评分)做值的rdd
ratingsAndPredictions = ratings.map(lambda r: ((r.user, r.product), r.rating)).join(predictions)
ratingsAndPredictions.first()
```
((506, 568), (5.0, 4.512168968112584))




```python
# MSE = ratingsAndPredictions.map(lambda ((user, product), (actual, predicted)): (actual - predicted) ** 2)\
# .reduce(lambda x,y: x + y) / ratingsAndPredictions.count()
# 用sum()和reduce(lambda x,y: x+y)是一样的
MSE = ratingsAndPredictions.map(lambda ((user, product), (actual, predicted)): (actual - predicted) ** 2).sum() \
/ ratingsAndPredictions.count()
print "Mean Squared Error =", MSE

import math
RMSE = math.sqrt(MSE)
print "Root Mean Squared Error =", RMSE
```
Mean Squared Error = 0.0839340560353
    Root Mean Squared Error = 0.28971374844



```python
# 计算APK(Average Precision at K metric)K值平均准确率
def avgPrecisionK(actual, predicted, k):
    predK = predicted[0:k]
    score = 0.0
    numHits = 0.0
    for i, p in enumerate(predK):
        if(p in actual):
            numHits += 1
            score += numHits / float(i + 1)
    if(len(actual) == 0):
        return 1.0
    else:
        return score / float(min(len(actual), k))
```


```python
actualMovies = [mu.product for mu in moviesForUser]
print actualMovies
```
[127, 475, 9, 50, 150, 276, 129, 100, 741, 1012]



```python
predictedMovies = [tkr.product for tkr in topKRecs]
print predictedMovies
```
[708, 482, 502, 603, 23, 182, 484, 479, 1020, 494]



```python
apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
print apk10
# 这里APK得分为0，表明模型在维该用户做相关电影预测上的表现并不理想
```
0.0



```python
import numpy as np
itemFactors = model.productFeatures().map(lambda (id, factor): factor).collect()
itemMatrix = np.array(itemFactors)
print itemMatrix.shape
```
(1682, 50)



```python
imBroadcast = sc.broadcast(itemMatrix)
```


```python
scoresForUser = model.userFeatures().map(lambda (userId, array): (userId, np.dot(imBroadcast.value, array)))
allRecs = scoresForUser.map(lambda (userId, scores): 
                            (userId, sorted(zip(np.arange(1, scores.size), scores), key=lambda x: x[1], reverse=True))
                           ).map(lambda (userId, sortedScores): (userId, np.array(sortedScores, dtype=int)[:,0]))
print allRecs.first()[0]
print allRecs.first()[1]
```
16
    [1274  453  135 ...,  631  412   96]



```python
# groupByKey返回(int, ResultIterable), 其中ResultIterable.data才是数据
userMovies = ratings.map(lambda r: (r.user, r.product)).groupByKey()
print userMovies.first()[0]
print userMovies.first()[1].data
```
2
    [237, 300, 100, 127, 285, 289, 304, 272, 278, 288, 286, 275, 302, 296, 292, 251, 50, 314, 297, 290, 312, 281, 13, 280, 303, 308, 307, 257, 316, 315, 301, 313, 279, 299, 298, 19, 277, 282, 111, 258, 295, 242, 283, 276, 1, 305, 14, 287, 291, 293, 294, 310, 309, 306, 25, 273, 10, 311, 269, 255, 284, 274]



```python
K = 10
MAPK = allRecs.join(userMovies).map(lambda (userId, (predicted, actual)):
                                    avgPrecisionK(actual.data, predicted, K)
                                   ).sum() / allRecs.count()
print "Mean Average Precision at K =", MAPK
```
Mean Average Precision at K = 0.024641131815



```python
# 使用MLlib内置的评估函数计算MSE,RMSE
from pyspark.mllib.evaluation import RegressionMetrics
predictedAndTrue = ratingsAndPredictions.map(lambda ((user, product), (predicted, actual)): (predicted, actual))
regressionMetrics = RegressionMetrics(predictedAndTrue)
print "Mean Squared Error =", regressionMetrics.meanSquaredError
print "Root Mean Squared Error =", regressionMetrics.rootMeanSquaredError
```
Mean Squared Error = 0.0839340560353
    Root Mean Squared Error = 0.28971374844



```python
# 使用MLlib内置的评估函数计算MAP, 它取所有物品来计算，不是取前K个，因此不用设定K值,故不叫MAPK
from pyspark.mllib.evaluation import RankingMetrics
predictedAndTrueForRanking = allRecs.join(userMovies).map(lambda (userId, (predicted, actual)):
                                                        (map(int, list(predicted)), actual.data))
rankingMetrics = RankingMetrics(predictedAndTrueForRanking)
print "Mean Average Precision =", rankingMetrics.meanAveragePrecision
```
Mean Average Precision = 0.0668399759999



```python
# 用我们自己实现的方法来计算MAPK，当K值较大时，结果同上面一样
K = 2000
MAPK2000 = allRecs.join(userMovies).map(lambda (userId, (predicted, actual)):
                                    avgPrecisionK(actual.data, predicted, K)
                                   ).sum() / allRecs.count()
print "Mean Average Precision at 2000 =", MAPK2000
```
Mean Average Precision at 2000 = 0.0668399759999

