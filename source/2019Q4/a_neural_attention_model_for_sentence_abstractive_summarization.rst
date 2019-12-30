=====================================================================
A Neural Attention Model for Sentence Abstractive Summarization
=====================================================================

Authors
--------

Alexander M.Rush, (现在哈佛任助理教授，主要做 data-driven的工作，把DL与structure-prediction 结合) 
Corpra Sumit, 
Weston Jason 

整个工作是facebook的。


个人评价
-----------

开坑之作。现在来看觉得训练数据很有问题：为啥用1st sentence作为输入就可以生成headline呢？ 这点站不住脚。



构建训练集
-----------

**核心方法**： 对Gigaword (Graff et al., 2003; Napoles et al., 2012), 将 headline 与 文章第一句 作为输入，即

:: 

    (first-sentence-of-article, headline)

得到 9.5 百万的输入；

**启发式过滤**：因为Gigaword contains mainly spurious headline-article pairs, 所以启发式地对以下情况做过滤：

1. 如果除去停用词就没有词了 (no non-stop-words)
2. 标题有 作者信息(byline) 或者 无关的编辑标记 (extraneous editing marks)
3. 有问题标题或者冒号

过滤后只有 4 万的输入；

**基础预处理**：

1. PTB tokenization
2. lower-casing
3. replace all digit with #
4. replace word-type seen less than 5 times with UNK


**词表统计**：

输入（first-sentence-of-article）: 119million tokens, 110K unique word-types (average 31.3 words per. sent)
摘要/标题(headline): 31 million tokens, 69K unique word-types (average 8.3 words per. headline)

**额外过滤**：

因为模型训练出来要在DUC-2004上做evalution，所以去掉了 DUC-2004 时间区间的文章。 （没说影响的文章数有多少）
