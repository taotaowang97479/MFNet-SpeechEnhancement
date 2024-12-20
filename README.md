# MFNet

This is the unofficial implementation of MFNet, from paper''a Mask Free Neural Network for Monaural Speech Enhancement''

arxiv:https://arxiv.org/abs/2306.04286

I appreciate the guidance and assistance from the author. After the correction following our discussion:<br>
1.The initial learning rate is 3e-4, correcting the value from 0.0034 in the paper.<br>
2.The features input to the network are compressed spectra, i.e., input = sign(stdct) * sqrt(abs(stdct)).<br>
3.DCT transformation without normalization.<br>
**I put the key code of STDCT, which may be useful for you.**

# Result

The model used the code modified by the author in the issues of his repo, [ioyy900205/MFNet#1](https://github.com/ioyy900205/MFNet/issues/1)

Experiments were conducted on the Voicebank+Demand, and the training parameters were set as follows:

- The warm-up strategy mentioned in the paper was not used
- Initial learning rate: 3e-4
- Batch size: 2
- Training for 100 epochs
- Cosine annealing with a period of 100 epochs
- Minimum value during cosine annealing: 1e-5

Performance of MFNet on the Voicebank+Demand test set:

|       | PESQ | STOI | CSIG | CBAK | COVL | SSNR |
| :---- | :--: | :--: | :--: | :--: | :--: | :--: |
| Noisy | 1.97 | 92.1 | 3.35 | 2.44 | 2.63 | 1.73 |
| MFNet | 3.05 | 94.6 | 4.19 | 3.55 | 3.63 | 9.79 |

The training process is as follows:

![MFNet_result](https://github.com/user-attachments/assets/aba7ac3d-6781-4607-886a-edfa49b77cc6)

