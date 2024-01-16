# MFNet

MFNet的非官方实现，来自论文''a Mask Free Neural Network for Monaural Speech Enhancement''

arxiv:https://arxiv.org/abs/2306.04286

非常感谢作者的指导和帮助，与作者交流过后的更正结果：<br>
1.初始学习率为3e-4，更正论文中的0.0034<br>
2.输入网络的特征为压缩谱，即input=sign(stdct) * sqrt(stdct)<br>
3.DCT变换不进行归一化<br>
**关于STDCT的关键代码已给出，希望对你有帮助**

# Result

本实验没有使用论文中的warm up策略，按照作者的建议训练：<br>
初始学习率为3e-4，训练1000个epoch，余弦退火周期也为1000，余弦退火最小值为1e-5<br>
MFNet在Voicebank+Demand(VCTK)测试集上的表现：

|       |  PESQ  | STOI  | SI-SNR  |
| :---: | :----: | :---: | :-----: |
| Noisy | 1.9799 | 92.11 | 8.4474  |
| MFNet | 3.0141 | 94.56 | 18.7835 |

另外：这是训练前100个epoch在测试集上的最佳结果。
