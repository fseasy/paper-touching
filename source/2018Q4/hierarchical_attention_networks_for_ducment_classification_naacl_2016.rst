=============================================================
Hierarchical Attention Networks for Document Classification
=============================================================

Influence
-----------

层次化建模文档的一个重要工作。引用数截止到2018年10月31日已经超过了500. 

常看到 HAN 结构，就是这篇论文里提到的模型，也就是 Hierarchical Attention Networks.

Motivation
-------------------

目前流行的神经网络没有考虑文档的结构信息，这篇文章试图验证猜想(直觉)：在模型里上加入文档结构的知识，能够帮助我们获得更好的表示。

    we test the hypothesis that better representations can be obtained by incorporating knowledge of document structure in the model architecture.

隐含在这个直觉之后的，是作者认为在回答一个问题时，文档中各部分不是相同程度地相关的；度量这种相关性，需要建模词之间的交互，而不仅仅是看其是否孤立地存在。 （这个好像跟 text-classification 没有关系？？）


Contributions
-------------------

论文的introduction提到，主要贡献就是新的神经网络结构： HAN.

具体来说

（1）模型有一个层次化的结构，来刻画文档的层次化结构，更符合直觉。

（2）基于层次化结构，在词级别和句子级别，都使用了Attention机制。使得模型在两个层次上有能力捕捉到对文档建模真正有用的信息。

