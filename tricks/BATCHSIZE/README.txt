https://zhuanlan.zhihu.com/p/64864995
【AI不惑境】学习率和batchsize如何影响模型的性能？

Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
表明通常当我们增加batchsize为原来的N倍时，要保证经过同样的样本后更新的权重相等，按照线性缩放规则，学习率应该增加为原来的N倍

On large-batch training for deep learning: Generalization gap and sharp minima
表明大的batchsize收敛到sharp minimum，而小的batchsize收敛到flat minimum，后者具有更好的泛化能力

Train longer, generalize better: closing the generalization gap in large batch training of neural networks
表明，大的batchsize性能下降是因为训练时间不够长，本质上并不少batchsize的问题，在同样的epochs下的参数更新变少了，因此需要更长的迭代次数。
当我们增加batchsize为原来的N倍时，如果要保证权重的方差不变，则学习率应该增加为原来的sqrt(N)倍

Don't decay the learning rate, increase the batch size
表明，衰减学习率可以通过增加batchsize来实现类似的效果，这实际上从SGD的权重更新式子就可以看出来两者确实是等价的，文中通过充分的实验验证了这一点。

A bayesian perspective on generalization and stochastic gradient descent
表明，对于一个固定的学习率，存在一个最优的batchsize能够最大化测试精度，这个batchsize和学习率以及训练集的大小正相关。