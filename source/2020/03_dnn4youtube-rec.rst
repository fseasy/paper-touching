======================================================
Deep Neural Networks for YouTube Recommendations
======================================================

Paul Covinton, Jay Adams, Emre Sargin, Google, Mountain View, CA

经典论文。2016年的文章了。

相关博客文章
~~~~~~~~~~~~~~~~~~~~

1.  `DNN YouTube Recommendations 召回 <https://zhuanlan.zhihu.com/p/42158565>`_

    该文章仅关注召回部分。

    **softmax 参数直接用item 的向量**
    
    在softmax阶段均采用图二中构造用户历史embedding序列时的与商品id对应embedding商品矩阵(即YouTubeNet中的视频矩阵)做内积计算。

    SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS

    **（负）采样方法**

    不是从列表页中，选择没有点击过的，而是：

    从当天所有点击过的商品中，除去当前用户点击过的商品，从剩余商品中随机选择 20 个商品作为负样本。


YouTube推荐时主要有3个挑战：

1. scale 大规模；高分布式学习与高效部署
2. Freshness 实时性；2个方面，一方面每秒都有大量新视频传上来；另一方面要尽快基于用户的近期action更新用户画像；
3. Noise 噪声，用户行为中有大量噪声

Google内部开始将基本上所有的learning-problem的通用解决范式，都迁移到深度学习模型。
该论文的模型也是如此。其有10亿参数，1000亿训练样本。

2. SYSTEM OVERVIEW 系统概览
===============================================

主要有2个网络，第一个网络用于候选生成，百万级别到百级别；第二个网络用于排序。

候选生成网络，输入是用户的历史行为，输出是几百个与用户相关的物料。论文里说 *The candidate generation network only provides broad personalization via collaborative filtering* , 这是说这个网络做的事情跟协同过滤本质相同？
另外，最后一句 *The similarity between users is expressed in terms of coarse features such as IDs of video watches, search query tokens and demographics.* , 表面是说用户相关性的表达通过这些粗糙的特征（观看的视频id序列，搜索query token，人口信息），
实际就是说这个网络中用户的实际特征是这些？

排序网络，用的特征更细细粒度，fine-level. 用了更多用户、视频的特征。做法是给每个视频打一个分，取top :math:`\rightarrow` 也就是点估计。

3.1 推荐作为分类
===============================================

推荐本来是做筛选，可以转化为打分——对每个item打分，然后取TOP；论文里说将其作为分类 `softmax` , 
其实也类似：毕竟 `softmax` 输出一个概率，也可以作为分数。

用户U，在上下文C下，是否要看视频 :math:`w_i`，用softmax


4.1 特征表示
=================================

Embedding Categorical Features 类别向量特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video-ID 这种词典特别大的，按点击频次排序取 TOP-N；

OOV 直接用 0向量 （没有用 UNK ）；

候选生成网络， 多个值得情况（如历史观看 videos ），直接取平均然后输入到网络；

不同特征域用到同一类特征，底层共享 emb： 比如 video-id 在 “曝光”，“最后一个观看 video”，“video ID that seeded the recommendation”，这里面用到的 video-id 对应的 emb，
底层都是1个 emb ： 好处是泛化，减少网络对内存的消耗
