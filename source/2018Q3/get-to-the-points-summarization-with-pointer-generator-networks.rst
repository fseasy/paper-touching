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


接下来看下 ``model.py``, 这个应该是我们核心要学习的。

::

    SummarizationModel

