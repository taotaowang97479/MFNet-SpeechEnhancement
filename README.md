# MFNet
''a Mask Free Neural Network for Monaural Speech Enhancement''

MFNet在Voicebank+Demand(VCTK)上的表现

与作者交流过后的更正结果：

1.初始学习率为3e-4，更正论文中的0.0034

2.输入网络的特征为压缩谱，即input=sign(stdct) * sqrt(stdct)
