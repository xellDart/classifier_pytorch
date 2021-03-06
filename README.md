# ShuffleNet v2 in Pytorch
- The paper about shufflenetv2: [shufflenet v2](https://arxiv.org/pdf/1807.11164.pdf)
- I implement the shufflenetv2, and test the performance on classification and detection tasks. You can use these codes to train model on your dataset.

## Version
- Python 3.6
- torch 1.1.0
- torchvision 0.3.0


#### Comparision with other models
As for classification task
batch_size=1, CPU

| Type | Acc | Time | MFLOPs |
| --- | --- | --- | --- |
| EfficientNet-B3 | 94.8 | 9.9 FPS | 1800 |
| ShuffleNet v2 | 94.7 | 48.3 FPS | 146 |
| MobileNet v2 | 94.3 | 30.5 FPS | 300 |
| MobileNet v3-Large | 89.8 | 29.7 FPS | 219 |
| MobileNet v3-Small | 90.9 | 45.0 FPS | 66 |

## Experiments
I train the model about 5 eopchs, and in each eopch, I test the performance of trained model.
```
Phase train loss: 0.6354673637662616, acc: 0.6564571428571429
Phase val loss: 0.5708242939949035, acc: 0.7146666666666667
Phase train loss: 0.493809922170639, acc: 0.7606285714285714
Phase val loss: 0.5668963393211365, acc: 0.724
Phase train loss: 0.4324655994551522, acc: 0.7994857142857142
Phase val loss: 0.4208303438186646, acc: 0.8048
Phase train loss: 0.38515312327657425, acc: 0.8273714285714285
Phase val loss: 0.37815397882064183, acc: 0.8298666666666666
Phase train loss: 0.3477836193084717, acc: 0.8467428571428571
Phase val loss: 0.34451772966384886, acc: 0.8441333333333333
```
And I didn't adjust any hyper parameters. After 15 epochs, the accuracy can reach 92.7%
```Phase val loss: 0.18356857439478239, acc: 0.9269333333333334```


