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

3.1 推荐作为分类

推荐本来是做筛选，可以转化为打分——对每个item打分，然后取TOP；论文里说将其作为分类 `softmax` , 其实也类似：毕竟 `softmax` 输出一个概率，也可以作为分数。

用户U，在上下文C下，是否要看视频 :math:`w_i`，用softmax
