Attention Is All Your Need
============================

关于Attention的定义
+++++++++++++++++++

Attention函数可以看做是一个映射，从 ``query && key-value对的集合 -> output 的映射`` 。其中 query, key, value, output 均是向量。

由 key & key 算出一个归一化的常量，然后用这个常量在对应的value上加权；加权后的结果相加，就是output . 

两种常见的attention
+++++++++++++++++++

一种是 ``additive attention`` (*Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 
Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.*), 
另一种是 dot-product(multiplicative) attention. 

尽管加法attention和点积attention有理论上相同的复杂度，但是点积attention在实际中更快，更省内存；因为它可以用矩阵乘法来高度优化！

当query和key的维度比较小时，加法attention与点积attention效果类似；但是当其维度较大时，加法attention就比不带scale的点积attention好！
猜测是因为当维度大时，**点积结果的magnitude很大，从而进入到softmax函数的极度低梯度的范围**。
所以，为了抵消这个影响，我们用 $ \sqrt d_k $ 来缩放这个magnitude.

    为什么magnitude会很大？ 是因为 query, key 都是均值为0，方差为1的独立随机变量，它们的点积 $ q \dot k = \sum_{i=1}^{d_k} q_i k_i $ 就是一个均值为0，方差为 的 $ d_k $ 的结果。

    这也解释了为什么要除以 $ \sqrt d_k $： 这样方差又成了1 ？？


Multi-Head Attention
++++++++++++++++++++++

传统的attention，就是在 query, key, value 上直接算一个加权结果。Multi-Head attention, 则是将用多个映射，分别将 query, key, value 单独映射为多个表示 ``[ (query_1, key_1, value_1), (query_2, key_2, value_2), ...]``，再在映射的结果上分别算attention，得到 ``[output1, output2, ...]``, 最后把这些output拼接后做一个映射，得到最终的output；

好处就是，允许模型联合地去关注输入的不同位置的不同子空间的表示。这个是单个的attention所不能做到的。

    其实还是好理解，就类比与 CNN 的多个filter吧。

实际种，用的一个矩阵来分别映射query，key，value； 而且，映射后的维度为原始维度除以head的数量，这样其实整体的运算量不变。

