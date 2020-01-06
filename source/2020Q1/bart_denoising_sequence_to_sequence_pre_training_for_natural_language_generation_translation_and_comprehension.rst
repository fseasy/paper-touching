=================================================================================================================
BART: Denosing Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
=================================================================================================================

优势
------

在判别式任务上与RoBERTa效果相似，但在生成任务上效果更好。


model
------

是一个 seq2seq结构（也就是Transformer），base设置下encoder/decoder 各6层； large设置下 encoder/decoer 各12层。

因为比BERT多了decoder对encoder的互动，整体比同规模BERT多10%的参数。


预训练模式
---------

方式：先corrupt数据，让模型尝试复原。优化这个reconstruction-loss.

BART允许任意的document corruption.

corrpution的方式有多种，分别是： token mask, token deletion, token infilling(span tokens replace with 1 mask),
sentence permutation, document rotation.


Finetuning 方式
----------------

分类任务： encoder, decoder都输入相同的内容，其中输入内容的最后一个token为"end"，
用这个token的decoder的final hidden state表示送给一个新的linear classifier.

Token分类任务：如SQuard, 同样用decoder的 final hidden state.

序列生成任务： BART有autoregressive的decoder，可以直接finetune.

探索使用BART来提升机器翻译的decoder
---------------------------------

前人有工作说明机器翻译可以通过预训练的encoder来提升效果，但是预训练的decoder作用不大。
这里尝试用整个BART来提升效果，主要是提升其他语言翻译为英语的效果。

核心思路就是把BART encoder中的embedding层给换掉，换成适配其他语言的embedding，
使用随机初始化。

训练分为两个步骤： 第一步 先固定住BART的大部分参数，然后主要训练embedding层，
position，第一层self-attention的projection。
第二步，用比较小的迭代轮次更新整个参数。

与其他训练目标比较
-----------------

因为发布的模型在训练数据、训练资源、模型结构、finetuning过程不同，因此
作者重新实现了很多的训练目标。用BERT来作为参照，因为BERT的设置和这里是一致的：
1M步，使用book (corpus)和 wikipedia数据。


与其他模型的区别
----------------

GPT只建模了左边的context，对某些任务是有问题的； ELMO分别建模了从左到右和从右到左，
但是彼此缺乏交互。

BERT不是auto-regressive，生成任务效果打了折扣。

unilm的预测是 条件独立的； BART避免了训练和预测的不同，因为BART的decode的输入永远是
没有corrupt的数据。

MASS比较像BART，它把输入侧的连续token mask掉，在decoder侧预测。
它在判别型任务上效果不好，因为encode和decoder的输入token集合是不想交的。

