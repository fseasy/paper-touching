======================================================
Deep Neural Networks for YouTube Recommendations
======================================================

Paul Covinton, Jay Adams, Emre Sargin, Google, Mountain View, CA

经典论文。2016年的文章了。

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

排序网络，用的特征更细细粒度，fine-level. 用了更多用户、视频的特征。做法是给每个视频打一个分，取top :math:`\leftarrow` 也就是点估计。


3.1 推荐作为分类
===============================================

推荐本来是做筛选，可以转化为打分——对每个item打分，然后取TOP；论文里说将其作为分类 `softmax` , 
其实也类似：毕竟 `softmax` 输出一个概率，也可以作为分数。

用户U，在上下文C下，是否要看视频 :math:`w_i`，用softmax
