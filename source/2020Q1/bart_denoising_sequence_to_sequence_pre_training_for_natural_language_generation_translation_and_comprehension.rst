=================================================================================================================
BART: Denosing Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
=================================================================================================================

model
------

是一个 seq2seq结构（也就是Transformer），base设置下encoder/decoder 各6层； large设置下 encoder/decoer 各12层。

因为比BERT多了decoder对encoder的互动，整体比同规模BERT多10%的参数。
