static environment:

训练经验:
1. Case 1 : with the training time steps increasing，reward and found targets increase until reaching 120 
targets at around 500 episodes, but after that, reward and found target go down. During this process, the loss keep increasing.
Reason: Learning ratio 1e-4 is too large, when reducing it to 1e-5, it converges pretty well, and doesn't go down anymore.
情况1 ： 随着训练次数的增加，奖励和找到的targets都会增加，直到训练了500个episodes后，达到120targets左右， 然后奖励和找到的目标都会下降，甚至忽高忽低，
比如有时一个episode能找到140个targets, 有时一个episode却只能找到9个targets。 在这个过程中，loss一直在上升。
原因 ：刚开始没有经验，考虑是不是过拟合的问题，应该不是过拟合而是欠拟合，学习率太大，在最小值的坑周围跳来跳去，落不下去。

2. Case 2 : 
