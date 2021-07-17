## 7.13 学习笔记

###  1.1    模型评估与检测

* 经验误差和过拟合 

学习器在训练集上的误差称为[训练误差]或[经验误差]，在新样本上的误差称为[泛化误差]。

过拟合：把训练样本自身的一些特点当做了所有潜在样本都会具有的一般性质，这样就会导致泛化性能下降

* 评估方法

  * 留出法

  直接将数据集D划分为两个互斥的集合，其中一个集合作为训练集S，另一个作为测试T

  * 交叉验证法

  现将数据集D划分为k个大小相似的互斥子集，k-1个子集的并集作为训练集，余下的子集作为测试集

  * 自助法

  给定一个m大小的样本，采样收集数据，每次从原始数据随机挑取，并放回式提取，重复执行m次，得到包含m个样本的数据集并作为训练集。

* 性能度量

  * 错误率和精度
  * 查准率、查全率与F1

* 比较检验

  * 后续写代码时实践总结

### 1.2   线性模型

- 学习内容
  - 基础学习内容：线性回归（最小二乘法，梯度下降法），逻辑回归，类别不平衡问题
  - 进阶学习内容：线性判别分析，多分类学习，岭回归，LASSO回归，ElasticNet回归
- 参考资料：西瓜书，网络博客资料
- 学习要求：做好学习笔记，及时上交。代码实现上述算法，使用sklearn.dataset中的load_diabetes糖尿病数据集进行算法的评价
- 备注：尽可能自己实现

### 2.1  模型描述

算法原理：

​	Training Set -> Learning Algorithm -> h

​	x -> h -> y

##### 损失函数

###### 1.直接求闭式解

模型h函数~Represent hypothesis：
$$
h_\theta(X)=\theta_0+\theta_1x
$$
参数~parameters：
$$
\theta_0,\theta_1
$$
代价函数~cost function：
$$
J(\theta_0,\theta_1)=\frac{\sum h_\theta(x^{(i)})-y^{(i)})^2}{2m}
$$
目标~Goal：
$$
minmizeJ(\theta_0,\theta_1)
$$

1. 属性只有一个时，计算出每个样本预测值和真实值之间的误差并求和，通过最小化均方误差，使用求偏导等于0，计算出两个参数

2. 原始线性回归不满足需求时，使用“广义线性回归”：
   $$
   y=g^{-1}(w^Tx+b)
   $$

```python
w = np.linalg.pinv(x).dot(y)
```



###### 2.梯度下降求解

Have a function J

want goal

outline:

keep changing \theta_0,\theta_1,until find goal 

终止条件，\theta 不再改变
$$
update\ rule:
\theta_1:=\theta_1-\alpha\frac{\vartheta}{\vartheta\theta_1}T(\theta_1)
$$
关键：求出代价函数的导数

【多变量线性回归】
$$
J(\theta_0,\theta_1,...)=\frac{\sum(h_\theta(x^{(i)})-y^{(i)})^2}{2m}
$$
为所有建模误差的平方和，求得偏导数





##### 梯度下降实践1-特征缩放

-帮助梯度下降算法更快收敛
$$
x_n=\frac{x_n-\mu_n}{s_n}\qquad \mu_n平均值，s_n标准差
$$

##### 梯度下降法实践2-学习率

$$
\alpha=0.01,0.03,0.1,...
$$

##### 算法大概思路

加载数据-》计算回归系数w-》绘制数据集-》绘制回归曲线和数据点

* 计算回归系数w：由公式推导，通过对目标函数求w的偏导数，得到
  $$
  w=\frac{\sum_{i=1}^{m}y_i(x^i-x')}{\sum_{i=1}^mx_i^2-x_ix'}
  $$
  用python实现：求和使用循环，如将此式子向量化，可转化为，一定是去掉均值之后，切记。
  
  
  $$
  x=(x_1,x_2,...)^T,x_d=(x_1-x',x_2-x',...)^T为去掉均值之后的x\\
  y=(y_1,y_2,...)^T,y_d=(y_1-y',...)为去掉均值之后的y\\
  \\  
  \\  
  w=\frac{x_d^Ty_d}{x_d^Tx_d}\\
  $$



 书写代码大概步骤：简单封装函数；导入数据集，处理数据集；测试模型，对比 sklearn



### 2.1  线性几率回归（逻辑回归）

将预测值投影到0-1之间，从而将线性回归问题转化为二分类问题

简单来说就是在线性回归的基础上增加了Sigmoid函数：
$$
f(z)=\frac{1}{1+e^{(-z)}}
$$


函数形状：

![](https://raw.githubusercontent.com/kristina100/Machine_Learning/master/picture/2.png)



然后将z替换为线性回归模型
$$
f(x)=f(w^Tx^*)=\frac{1}{1+e^{-(w^Tx^*)}}
$$
由于是二分类问题，可看作如下的概率模型：
$$
P(y=1|x)=f(x)\\
P(y=0|x)=1-f(x)
$$
于是逻辑回归可以通过极大拟然估计的方法求解：
$$
ln{\frac{p(y=1|x)}{p(y=0|x)}}=wx^T+b\\
极大拟然法估计：经过数值优化推导得出了l(\beta)=\sum_{i=1}^m(-y_i\beta^Tx_i+ln(1+e^{\beta^Tx_i}))
\\
再对w求导=》w:=w-\eta\frac{\partial L}{\partial w}
$$



### 2.2  类别不平衡问题

就是指分类任务中不同类别的训练样例数目差别很大的情况。

对于平衡的数据，一般都用准确率，也就是（1-误分率）作为一般的评估标准。这种标准的默认假设前提是：“数据是平衡的，正例与反例的重要性一样，二分类器的阈值是0.5。”在这种情况下，用准确率来对分类器进行评估是合理的。

但是当类别不平衡的时候，意义不大。

提升不平衡分类准确率的问题的方法：采样，阈值移动，调整代价或权重。

![](https://raw.githubusercontent.com/kristina100/Machine_Learning/master/picture/1.png)



## 进阶

### 1.1  岭回归 和 Lasso回归

其实两者就是在标准线性回归的基础上分别加入L 1，L 2正则化，也就是修改cost function

Lasso回归如下：
$$
J=\frac{\sum_{i=1}^n(f(x_i)-y_i)^2}{n}+\lambda||w||_1
$$
Ridge回归如下：
$$
J=\frac{\sum_{i=1}^n(f(x_i)-y_i)^2}{n}+\lambda||w||_2^2
$$
用数学公式可以证明等价如下：
$$
f(w)=\sum_{i=1}^m(y_i-x_i^Tw)^2\qquad
s.t.\sum_{i=1}^nw_j^2\leq t
$$
将岭回归系数用矩阵形式表示：
$$
w=(X^TX+\lambda I)^{-1}X^Ty
$$
也就是将X^TX加上一个单位矩阵变成非奇异矩阵并进行求逆运算。

岭回归的几何意义：

以两个变量为例, 残差平方和可以表示为 ![[公式]](https://www.zhihu.com/equation?tex=w_1%2C+w_2) 的一个二次函数，是一个在三维空间中的抛物面，用等值线来表示。而限制条件 ![[公式]](https://www.zhihu.com/equation?tex=w_1%5E2+%2B+w_2%5E2+%3C+t) ， 相当于在二维平面的一个圆。这个时候等值线与圆相切的点便是在约束条件下的最优点，如下图所示：



![](https://raw.githubusercontent.com/kristina100/Machine_Learning/master/picture/3.png)





Lasso的惩罚项：
$$
\sum_{i=1}^{n}|w_i|\leq t
$$


Lasso的几何解释：

以两个变量为例，标准线性回归的cost function还是用二维平面的等值线表示，而约束条件则与岭回归的圆不同，LASSO的约束条件用方形表示，如下图:

![](https://raw.githubusercontent.com/kristina100/Machine_Learning/master/picture/4.png)

相比圆，方形的顶点更容易与抛物面相交，顶点意味着对应多个系数为零，可以很好的筛选变量。

代码实现：就是在线性回归的基础上增加正则化

直接放在第一次写的代码中。

### 1.2   线性判别分析

将训练样本投影到一条直线上，使得同类的样例尽可能近，不同类样例尽可能远。即：让各类协方差尽可能小，不同类之间的中心距离尽可能大。

* 类内散度矩阵
  $$
  S_w=\sum_{x\in X_0}(x-\mu_0)(x-\mu_0)^T+\sum_{x\in X_1}(x-\mu_1)(x-\mu_1)^T
  $$

* 类间散度矩阵
  $$
  S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T
  $$

* 广义瑞利商
  $$
  J=\frac{w^TS_bw}{w^TS_ww}
  $$

J越大越好。



### 1.3   多分类学习

基本思路：将多分类问题拆解为若干个二分类任务求解；根本问题仍然是二分类问题

* **O V O **   假如某个分类中有N个类别，将这N个类别进行两两配对（两两配对后转化为二分类问题）。那么可以得到![img](https://img-blog.csdn.net/20170901162649098)个二分类器。在测试阶段，把新样本交给这![img](https://img-blog.csdn.net/20170901162649098)个二分类器。于是可以得到![img](https://img-blog.csdn.net/20170901162649098)个分类结果。把预测的最多的类别作为预测的结果。

* **O V R**   每次将一个类别作为正类，其余类别作为负类。此时共有（N个分类器）。在测试的时候若仅有一个分类器预测为正类，则对应的类别标记为最终的分类结果

* O v O的优点是，在类别很多时，训练时间要比OvR少。缺点是，分类器个数多。

  O v R的优点是，分类器个数少，存储开销和测试时间比OvO少。缺点是，类别很多时，训练时间长。

![](https://raw.githubusercontent.com/kristina100/Machine_Learning/master/picture/8.jpg)

#### *EOOC技术

编码：对N个类别做M次划分，每次划分将一部分类别划分为正类，一部分划分为反类，从而形成一个二分类训练集，一共产生M个训练集。

解码：M个分类器分别对测试样本进行预测，预测标记再组成一个编码，将这个编码与每个类别的编码比较，返回距离最小的类别作为最终预测结果