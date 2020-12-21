##############################################################################
Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba
##############################################################################

作者： jizhe Wang, Pipei Huang（黄丕培）. （见 `万物皆向量——双十一淘宝首页个性化推荐背后的秘密 <https://www.infoq.cn/article/dtlzivn21zhlxykycxua>`_）

大名鼎鼎的 EGES. 淘宝主要面临3个挑战： scalability, sparsity, cold-start. 这篇文章就要解决这3个问题。

GMV: Gross Merchandise Volume （总成交额）

Double-Eleven Day: 双11

Taobao： 

是 C2C 平台；淘宝有 10 亿用户， 20 亿 items (commodities). 贡献阿里 75% 的流量；

最主要的问题，让用户快速找到需要的、感兴趣的商品！ 推荐是其中的关键技术。 手机淘宝主页基于用户历史行为生成，占据推荐流量的40%； 推荐贡献了最要的营收(revenue)和流量。

Scalability: 在 10 亿用户 20 亿物品前，很多推荐方法不好使；

Sparsity: 用户只与部分 item 有行为。

Cold-Start: 每小时，有百万级别的新 item 被上传到系统；这些 item 没有用户行为；处理这些物料，或者将这些物料推荐给 用户，是一个困难的事情。

使用 matcing -> ranking 二阶段策略；
matching 是给用户有行为的 item，生成1个相似 item 候选；
ranking 基于用户的偏好，用 DNN 排序候选 commodities.
每阶段面对不同的问题，有不同的技术解决方式。

这篇文章主要关注 matching 阶段！

CF 的方法，主要考虑用户历史中，物品间的 co-occurence. 这里用 item 图上的 random-walk, 可以 items 间的高阶相似； 这被称为 BGE (Base Graph Embedding).

BGE 比 CF 强，但是依然解决不了只有很少行为甚至没有行为的 item；为了解决这个问题，因而提出了利用 side-information 提升 Emb 生成的方式，即 GES (Graph Embedding with Side information)。

又，淘宝的商品有很多种类型的 side information, 例如 类别、品牌、价格等； 经验性地，不同类型的 side information 对 commodities emb 贡献不同，因此有必要在 side information 上考虑加权； 这种模型被称为 EGES (Enhanced Graph Embedding with Side information)

总结，在 matching 阶段，有 3 个重点工作：

1. 基于用户行为，构建 item graph
2. 学习 item embedding (EGES, GES 优于 BGE)
3. 部署 graph embedding system.  (在自研的 Xensorflow 上)

==========================
构建 item graph
==========================

图是 加权有向 图；

1. 将用户连续的行为，拆分为 session-based behaviour.

    将连续的用户行为，按 1 个小时为 window 拆分为多个 session.

    Why: a. 如果不拆分，序列太长，计算、存储开销大
    b. 用户的兴趣随时间有漂移

    其实这里觉得细节还不是太清楚——
    
    拆分为多个 session，应该是每个 session 都会用吧？而不是只用最近1个小时的？
    - 如果是每个 session 都用，那这个相比用整个行为序列，也不能减少多少计算、存储开销？
    - 但只有那个最后1个 session，想想还是不太可能……
    - 应该是用全不的 session； 
    - 切分为 session，相比不切，按照后面的构建边的方式，其实就是少了跨 session 的边；
    这种量，可能积累起来还是比较多的吧——毕竟会有很多个 1 个小时切分；

    拆分的起始时间是统一的，还是基于每个用户的行为序列的起始时间？或者是结束时间（当前时间倒推）？
    - 应该是基于每个行为序列的起始时间？相比倒推，这样能够保证每次计算都是一致的结果；
    - 其实时间统一，不太现实，也没必要？

2. 基于 session-based behaviour, 在 图中连有向边。 如 序列是 A D B. 则连边 A -> D; D -> B;

3. 计算每条边的权重： 就是这条边被连了多少次。 
    
    这样，图上的每条边，其实是全部用户行为的表现。

需要过滤掉一些噪声数据：

1. stay after click less than 1s
2. spam user (3 个月内买 1,000 个 item，或者点了超过 3,500 的用户)
3. item 的 id 没变，但是零售商把其内容大幅改变了

==========================
计算 emb
==========================

方法1： BGE
+++++++++++++++++++++++

就是利用 DeepWalk 方式来构建 Emb；

具体地：

1. 随机游走，得到 item 序列 数据集

    随机游走方式：论文里没有详细描述，只说了
    
    - 从1个节点到另一个节点的概率：

    .. math::

        p(v_j | v_i)) = \begin{cases}
            \frac {M_{i, j}} {\sum_{j\in N_+{(v_i)}} M_{i,j}}, & v_j \in N_+(v_i) \\
            0, & e_{ij} \notin E
        \end{cases}

        很简单，就是有边的就按归一化权重走；没边的不走；
    
    但是
    
    1. 如何选起始节点
    2. 路径长度
    3. 总共生成多少个

    没有细说。

2. 得到 item 的序列 数据集后，直接用 word2vec 的 Skip-Gram + negative-sampling
(图里面画的是 smapled softmax, 可能差别没有那么大)模式训练即可。

方式2： GES
+++++++++++++++++++++++

BGE 没法处理冷启动问题；需要考虑 side information.

编码也非常简单——每个 side-information 和 item id 一样的等同对待，one-hot => 映射为 d 维的向量 => avg pooling, 就得到了 item 的表示向量。 

训练方式依然保持不变。


方式3： EGES
+++++++++++++++++++++++

BGE 只是简单的把各个 side-information 平均起来，一个简单的优化就是加权求和。

论文的权重计算也非常简单，就是对每个 side-information, 
学习 1 个权重值，然后整体归一化一下，加权求和。 

加权求和用的是 `softmax` , 论文里竟然没有点明，自己竟然也看了半天论文里说的为啥要用 ..math:`e^{a_{v}^{j}}` = =

此外，论文引出该方法时的说法我觉得值得商榷：

    For example, a user who has bought an IPhone tends to view Macbook or IPad because of the brand "Apple", while a user may buy clothes 
    of different brands in the same shop of taobao for 
    convenience and lower price.

然后为了解决这个问题，所以对不同的side-information做加权。

但是，这个加权其实是全局的，即对任何的物料，权重都是一样的；然而上面说法，更合理的应对方法，
应该是针对不同类型的物料, 或者不同的用户，side-information 的权重应该不同。当然，
不同的用户用不同的权重，这显然在这里不太现实。

因而反过来说，论文里的这个引子，还是不够好。


===========================
实验
===========================

用于验证效果的方法：

1. link prediction task (offline Evaluation)
2. online experimental result on Model Taobao App. 
3. some real-world cases

link prediction task (offline Eval)
+++++++++++++++++++++++++++++++++++++++++++++++

link prediction 是网络中的基础问题，所以用作离线实验。

任务定义： 从图中，随机抹掉一些边，然后预测边是否存在。

细节： 1. 1/3的边被随机抹掉，作为测试集；剩余的边作为训练集； 2. 



==========================
其他
==========================

`一天造出10亿个淘宝首页，阿里工程师如何实现？ <https://my.oschina.net/u/4662964/blog/4743526>`_
