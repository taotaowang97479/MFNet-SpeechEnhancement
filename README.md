# MFNet

This is the unofficial implementation of MFNet, from paper''a Mask Free Neural Network for Monaural Speech Enhancement''

arxiv:https://arxiv.org/abs/2306.04286

I appreciate the guidance and assistance from the author. After the correction following our discussion:<br>
1.The initial learning rate is 3e-4, correcting the value from 0.0034 in the paper.<br>
2.The features input to the network are compressed spectra, i.e., input = sign(stdct) * sqrt(stdct).

# Result

This experiment did not utilize the warm-up strategy mentioned in the paper. Instead, following the author's recommendation, the training parameters were set as follows:

- Initial learning rate: 3e-4
- Training for 1000 epochs
- Cosine annealing with a period of 1000 epochs
- Minimum value during cosine annealing: 1e-5

Performance of MFNet on the Voicebank+Demand (VCTK) test set:

|       |  PESQ  | STOI  | SI-SNR  |
| :---: | :----: | :---: | :-----: |
| Noisy | 1.9799 | 92.11 | 8.4474  |
| MFNet | 3.0141 | 94.56 | 18.7835 |

Additionally, these are the best results on the test set obtained during the first 100 epochs of training.
