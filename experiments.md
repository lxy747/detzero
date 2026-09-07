
## CenterPoint，1sweeps，bench2drive  
用DetZero CenterPoint模型，1sweeps，模型参数未改。数据集用Bench2drive，有4.7w。  
类别：['car','van','truck','bicycle','traffic_cone','pedestrian', 'generic_object']  
```
mAP: 0.7602
mATE: 0.1829
mASE: 0.2198
mAOE: 0.1967
mAVE: 4.1407
mAAE: 1.0000
NDS: 0.6201
Eval time: 26.2s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car             0.838   0.089   0.078   0.065   4.385   1.000
van             0.000   1.000   1.000   1.000   1.000   1.000
truck           0.962   0.075   0.034   0.017   8.515   1.000
bicycle         0.970   0.024   0.028   0.031   3.280   1.000
traffic_cone    0.972   0.024   0.166   nan     nan     nan
pedestrian      0.830   0.036   0.015   0.054   7.665   1.000
generic_object  0.749   0.032   0.218   0.012   0.000   1.000
```
