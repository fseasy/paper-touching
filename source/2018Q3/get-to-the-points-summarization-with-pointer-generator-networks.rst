==================================================================
Get-to-the-point: Summarization with pointer-generator networks
==================================================================

Attr
=======

author
--------

See Abigail
    Google Brain & stanford NLP

Run code
---------

repo: https://github.com/abisee/pointer-generator 

1. download data

作者给出了完整的数据处理流程，放在 https://github.com/abisee/cnn-dailymail 下。

不过我们使用一个用户（见讨论 https://github.com/abisee/cnn-dailymail/issues/9 ）提供的已经处理完的数据，结果在 https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail 

下载其中的 *FINISHED_FILES* 对应的链接里的文件，得到一个zip，解压后是如下的结构：

::

    chunked/
        test_000.bin
        ...
        test_011.bin
        train_000.bin
        ...
        train_287.bin
        val_000.bin
        ...
        val_013.bin
    test.bin
    train.bin
    val.bin
    vocab

其中 ``chunked`` 存储的是将外边对应的文件按照1K每份做划分后的结果——划分是为了后面训练时多进程读取输入。

*\*.bin* 是使用 ``abisee/cnn-dailymail`` 仓库中的 ``make_datafiles.py`` 脚本处理原生txt的结果。
处理过程大概是用 ``tensorflow.core.example.example_pb2.Example`` 
对象来存储tokenize过的article和abstract,
然后序列化这个对象，
并以二进制方式写入到文件。则应该是tf标准的处理方式。暂时先不管。

2. 安装tf

还是觉得在公司电脑上运行，所以登上去用virtualenv 装了tf1.2；但是看README是1.2.1，不知道有没有差别；

此外，需要重新把文件下载到服务器上。好麻烦……


3. 运行

搞定。用tf1.2顺利把train跑起来了。还通过tensorbord看了下loss（scalar）；不过看graph和embedding都失败了。
应该是没有把graph保存下来；embedding查看则一直在parsing，不知道是不是工作用的Mac air性能太差了。

不管怎么说，前期运行已经OK了。下面就是看代码了！

Reading Code
-------------

主入口是 ``run_summarization.py``
++++++++++++++++++++++++++++++++++

::

    part1: config-argument definitions
    part2: global function
        a. calc_running_avg_loss
            平滑loss；每一步将之前的累积loss decay，同时加上缩放过的当前一步的loss，得到新的累计loss
            以此新的累计loss为当前的loss
        b. restore_best_model
            将eval的模型拷贝到train下；其中用到了tf的参数恢复的能力！
        c. convert_to_coverage_model
            转换模型
        d. setup_training
            设置training & 跑training； 使用了tf.train.Supervisor
        e. run_training
        f. run_eval
        g. main
            判定是解码时，设置 batch-size = beam-size. 利用是
            in decode mode, we decode one example at a time. On each step, we have 
            beam_size-many hypotheses in the beam, 
            so we need to make a batch of these hypotheses.
            应该得结合beam-search怎么做的来看了。

            用一个namedtuple来存储flags的解析的且需要的东西

            用 tf.set_random_seed 来设置随机数种子

            在decode模式下可以看到：将 max_dec_steps 强制设为1了；解释说是每次做一步！然后调用的是
            BeamSearchDecoder.decode 来做的。


接下来看下 ``model.py``
+++++++++++++++++++++++++++++++

这个应该是我们核心要学习的。

::

    SummarizationModel

        _add_placeholders
            输入部分
            enc_batch, int32, (batch-size, None), 说明每次每个batch的size部署固定的！
            enc_len, int32, (batch-size), batch中每个句子的长度？
            enc_padding_mask, float32, (batch-size, None) mask 处理！
            如果是 pointer-gen 网络
                enc_batch_extend_vocab, int32, (batch-size, None) 每个instance的扩展vocab
                max_art_oovs, int32, [], 不清楚是什么
            
            dec_batch, int32, (batch-size, None)
            target_batch, int32, (batch-size, None)
            dec_padding_mask, float32, (batch-size, None)
        
            decode & coverage时
                prev_coverage, float32, (batch-size, None)
        
        _add_encoder
            创建lstm cell，用的是
                tf.contrib.rnn.LSTMCell
            这个和 tf.nn.rnn_cell.LSTMCell 是等价的（alais）; initializer 用的是同一个initializer:
                self.rand_unif_init = tf.random_uniform_initializer(
                    - rand_unif_init_mag, rand_unif_init_mag, seed=123
                )
                state_is_tuple = True, 默认行为；返回的结果有

            创建双向LSTM网络，用的是
                tf.nn.bidirectional_dynamic_rnn
            传入了 sequence_len，这个应该是来自data部分的输入；
            swap_memory = True, 这个参数的解释是
            Transparently swap the tensors produced in forward inference but needed for back prop 
            from GPU to CPU. This allows training RNNs which would typically 
            not fit on a single GPU, with very minimal (or no) performance penalty.
            似乎是多GPU时把这个参数打开；默认是False的；
            

``data.py`` 模块处理数据
+++++++++++++++++++++++

比较有意思的：

在从外部字典文件加载到内部字典时，会把最后一个加入的词打出来——方便定位，很细心。


如下是特殊字符：

:: 

    # <s> and </s> are used in the data files to segment the abstracts into sentences. 
    # They don't receive vocab ids.
    SENTENCE_START = '<s>'
    SENTENCE_END = '</s>'

    PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
    UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
    START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
    STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


包含一个用于TensorBoard可视化的函数。

有一个函数来生成 ``tf.Example`` , 其中用 ``glob.glob`` 函数来扩展通配符，这个挺不错的。从data中读取tf.Example
时，用了 `struct` 这个标准库，这个是用来以二进制方式在C类型与Python类型做交换的；不是特别懂这个，可能算是序列化的
一种方式？用 ``struct.unpack`` 来完成的。

原来，虽然在generator机制下，input中的UNK（不出现在全局词典中）会被拿出来作为额外的字典；
但是训练语料中abstract的内容中，可能仍然含有不在全局词典 + input额外词典中的词，所以 **训练语料的输入中，还是有可能有UNK！**



