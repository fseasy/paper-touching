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

尽管加法attention和点积attention有理论上相同的复杂度，但是点积attention在实际种更快，更省内存；因为它可以用矩阵乘法来高度优化！

当query和key的维度比较小时，加法attention与点积attention效果类似；但是当其维度较大时，加法attention就比不带scale的点积attention好！
猜测是因为当维度大时，点积结果的magnitude很大，从而进入到softmax函数的极度低梯度的范围。
所以，为了抵消这个影响，我们用 $ \sqrt d_k $ 来缩放这个magnitude.