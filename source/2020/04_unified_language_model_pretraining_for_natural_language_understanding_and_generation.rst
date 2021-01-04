=========================================================================================
Unified Language Model Pre-training for Natural Language Understanding And Generation
=========================================================================================


预训练设置
----------

1. 一个batch里，Bidirectional LM, Unidirectional LM, Seq2Seq LM 的目标各占 1 / 3.

2. 模型结构和BERT-large相同，与GPT一样，使用gelu激活函数；24-layer, 1024 hidden size, 16 attention heads, 大约340M参数。
    基于 BERT-large 初始化

3. 数据上，使用English Wikipedia 和 BookCorpus. (跟BERT一样)

4. 输入序列最大长度是512 （看起来后面的finetune阶段更长一些，为788！）

4. mask策略上，15%的概率做mask：
    做mask的时候，80%概率mask token为 ``[MASK]`` , 10%随机替换，10%保持不变； 
    替换为mask时，80%替换unigram，20 bigram或trigram.

5. Adam, beta1 = 0.9, beta2 = 0.99; lr = 3e-5; 
    linear warm-up at first 4W steps, linear decay; 
    dropout-rate = 0.1; weight-decay = 0.01！
    batch-size = 330;
    total steps = 77W steps, 7hours/1W steps, 所以一共要 77 * 7 = 539 h = 22 days；
    8张 V100 32G, 混合精度训练

**疑问：**

1. BERT-large使用什么激活函数？ => 使用的也是 GeLU(Guassian Error Linear Unit)

2. mask的具体策略是什么？是依次针对每个token做mask吗？

3. 训练 seq2seq LM，segment1, segment2 是如何选的呢？ 难道就是相邻的两个句子吗？


Finetuning 阶段
-------------------

对NLU而言，Finetune的设置与BERT相同；以分类为例，就是把[SOS] (也就是BERT里的[CLS])表示再接一个softmax layer.
更新预训练部分和softmax部分。

对NLG而言，Finetune与Pre-training是类似的！（并没有使用teacher-forcing这样的训练方式）； 将输入准备为
[SOS]S1[EOS]S2[EOS]格式，然后对S2部分和S2后面的[EOS]做随机替换，让模型去预测[MASK].

**疑问**：

对NLG，解码的时候该怎么做呢？不是auto-regressive的，难道是解码一个token就要把前面的input全部输入进去？


实验：Abstracitve Summarization任务
---------------------------------------

选取了 CNN/DM 和 Gigaword 数据集。

按照 seq2seq LM 的 finetune方式来finetune： 将 document 作为segment1, 将 summary 作为segment2.
对**合并后的输入**(需要确认)做最大长度截断。

在训练集上finetune. 大部分参数复用预训练阶段；mask概率改为0.7 (之前只有0.15). 使用 label-smoothing with rate == 0.1.

CNN/DM上，batch-size = 32, max-length = 768;

Gigaword, batch-size = 64, max-length = 256; 

解码的时候：
    beam-size = 5； 去除重复的trigram, 基于dev调整长度
    CNN/DM, 输入文档被truncate为前640个token；
    Gigaword 输入被truncate为前面192个


实验：Generative QA
------------------------

CoQA上，将对话历史、问题、passage拼接起来作为segment1； 

如果输入超过了470长度，那么将passage在一个滑动窗口内切分几个chunk，取与question word-overlap最大的chunk作为输入。

实验：用Generative QA自动生成问题，再用自动生成的问题反过来优化QA任务
------------------------------------------------------------------------------

这个设计还是有点意思的；另外，还提到一句说“在数据增强的实验中，用的是bidirectional masked language model作为
辅助任务，而不是直接用（端到端的seq2seq训练？）”，这样可以获得2.3个点的绝对提升，猜测是可以避免在增强数据上灾难遗忘。


代码阅读
-------------


**decode_seq2seq.py**

max_seq_len 至少应该比tgt_seq_len 大2； 应该是因为seq input有额外的token；


