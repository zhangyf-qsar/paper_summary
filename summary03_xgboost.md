```html
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```

# 论文阅读笔记XGBoost: A Scalable Tree Boosting System 



## 1. 梯度提升树简介

### 1.1 正则化的目标函数

​	根据加性模型原理，XGBoost的预测值是若干个树模型预测值的累加求和
$$
\hat{y}_{i}=\phi(x_i)=\sum_{k=1}^{K}f_k(x_i), f_k \in F \tag{1}
$$
其中$F=\{f(x) = w_{q(x)}\}$表示所有决策树组成的空间。$q$表示某个树结构，它将样本$x$映射到对应的叶子节点，$q_{x}$就表示这个样本对应叶子节点的序号，$w_{q(x)}$是这个叶子节点的权值。$T$是树中叶子节点的个数。每个$f_k$都与一个独立的树结构$q$以及叶子节点权值$w$对应。

​		要优化的目标函数为
$$
\begin{equation}
\mathcal{L}(\phi) = \sum_{i}l(\hat{y}_i, y_i) + \sum_{k}\Omega(f_k) \\
where \space\Omega(f)=\gamma T + \frac{1}{2}\lambda\|w\|^2  \tag{2}
\end{equation}
$$
其中$l$是可微分并满足凸函数性质的损失函数，正则项$\Omega$限制模型的复杂性

### 1.2 梯度提升树

​		损失函数无法使用传统优化方法，需要应用叠加式方法。例如，当构建第$t$个树(即加性模型第$t$次迭代)时，假定前$t-1$颗树已经构建完成，此时损失函数为
$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n}l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)
$$
这表明我们只需寻找$f_t$，在$\hat{y}_i^{(t-1)}$已知的条件下，使预测结果达到最好即可。

应用二阶泰勒展开，得到
$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n} \left[ l(y_i, \hat{y}^{(t-1)}) 
 + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
$$
其中$g_i$和$h_i$分别是损失函数的一阶导数和二阶导数
$$
g_i = \partial_{\hat{y}^{(t-1)}}l(y_i, \hat{y}^{(t-1)}) \\
h_i = \partial^2_{\hat{y}^{(t-1)}}l(y_i, \hat{y}^{(t-1)})
$$
省略常数项，简化为
$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right]  + \Omega(f_t)  \tag{3}
$$
展开$\Omega$
$$
\mathcal{L}^{(t)} = \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right]  + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T}w_j^2
$$
上式是对每个样本求和，将其改写为先对叶子节点求和，在每个叶子节点内部，再对样本求和
$$
\mathcal{L}^{(t)} = \sum_j^T \left[ (\sum_{i \in I_j}g_i)w_j 
+ \frac{1}{2} (\sum_{i \in I_j}h_i+\lambda)w_j^2 \right] + \gamma T \tag{4}
$$
其中$I_j=\{i|q(x_i)=j\}$表示属于叶子节点$j$的所有样本。根据一元二次函数极值公式，可以求得$w_j$的最优值为
$$
w_j^* = -\frac{\sum_{i \in I_j}g_i}{ \sum_{i \in I_j} h_i + \lambda} \tag{5}
$$
损失函数最优值为
$$
\mathcal{L}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^T 
\frac{(\sum_{i \in I_j} g_i) ^ 2} {\sum_{i \in I_j} h_i + \lambda} + \gamma T \tag{6}
$$
在构建决策树分裂节点时，为了使损失函数降低最大，重新定义信息增益函数为
$$
\mathcal{L}_{split} = \frac{1}{2} \left[ 
\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} + \lambda} 
+ \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} + \lambda} 
- \frac{(\sum_{i \in I}g_i)^2}{\sum_{i \in I}h_i + \lambda}
\right] -\gamma \tag{7}
$$
它表示左子树与右子树得分之和与根节点得分的差，节点分裂时应让这个数值尽可能的大。

### 1.3 Shrinkage和特征采样(列采样)

​		Shrinkage表示对每颗树的预测结果累加求和时，增加一个缩放因子$\eta$，类似于梯度下降算法的学习率，它降低了单颗树对预测结果的影响，为接下来要构建的树保留了更多的提升空间。

​		特征采样是指不使用全部特征，而是使用全部特征的子集。特征采样防止过拟合并加快计算速度



## 2 分裂点查找算法

### 2.1 精确的贪心算法（Exact Greedy Algorithm）

​		遍历所有特征所有可能的分裂点。计算时需要对特征按数值大小排序，并计算每个分裂点的得分(由公式(7)定义)。

### 2.2 近似算法和Weighted Quantile Sketch

​		使用预先定义的分位数作为分裂点。分位数由参数$\epsilon$确定，在文章中被称为近似因子，它表示大约将选取$1/\epsilon$个候选的分裂点。与普通分位数不同，XGBoost在计算特征分位数时，对于每个样本，使用了其损失函数的二阶导数$h$作为权重。为了方便对其进行数学描述，首先定义rank函数
$$
r_k(z) = \frac{1}{\sum_{(x,h) \in D_k} h} \sum_{(x,h) \in D_k, x<z}h \tag{8}
$$
rank函数表示了属性值小于$z$的样本所占的比例，只不过这个比例不是简单计数，而是使用了$h$加权。目标是找到候选的分裂点$\{s_{k1}, s_{k2}, ..., s_{kl}\}$，使得
$$
| r_k(s_{k, j} - r_k(s_{k, j+1}))|< \epsilon \tag{9}
$$
也就是说，任意两个分裂点之间的样本比例不超过$\epsilon$。

​		之所以使用$h$加权，可以将公式(3)改写为
$$
\sum_{i=1}^n \frac{1}{2} h_i \left[ f_t(x_i) - (- \frac{g_i}{h_i}) \right]^2 
+ \Omega(f_t) + constant
$$
注意原文可能存在符号错误(参考：https://datascience.stackexchange.com/questions/10997/need-help-understanding-xgboosts-approximate-split-points-proposal)

如果以$-\frac{g_i}{h_I}$为标签，$\frac{1}{2} \left[ f_t(x_i) - (- \frac{g_i}{h_i}) \right]^2$就是平方损失函数，$\frac{1}{2} h_i \left[ f_t(x_i) - (- \frac{g_i}{h_i}) \right]^2$就是$h$加权的平方损失函数。对于权重一致的情况下，已经存在了quantile sketch算法。然而，还没有哪个现成的分位数略图算法可以解决有权重的样本。为此，作者提出了一个新的有理论支撑的分布式加权分位数略图算法(Weighted Quantile Sketch)，支持融合与剪枝操作，而每一个合并与剪枝操作又能保证一定量级上的准确度。

### 2.3 如何划分有缺失值的属性

​		对于缺失值处理，在节点划分时，XGBoost提出"default direction"的概念，尝试将缺失值数据分别划分到每个子节点，损失函数降低最多时，这个子节点就是最优的"default direction"。这相当于选取了损失函数降低最多的划分。在原文中该算法描述为Algorithm 3 。
