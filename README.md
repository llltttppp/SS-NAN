# SS-NAN
Keras implementation for the CVPR 2017 workshop paper [Self-Supervised Neural Aggregation Networks for Human Parsing](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w19/papers/Zhao_Self-Supervised_Neural_Aggregation_CVPR_2017_paper.pdf)

This code implement three kinds model for human parsing dataset LIP [Look into Person: Self-supervised Structure-sensitive Learning and A New Benchmark for Human Parsing](https://arxiv.org/abs/1703.05446)

## Results:
Pixel Accuracy: 85.8% 
MeanAccuracy:   58.1%   
MeanIoU:        47.90%

## Requirments:
keras 2.0.9
tensorflow 1.3.0
python 3.5.4
* Anaconda=5 (not neccessary just for convenience)

## Data Preparation

Please download the LIP(single person train, val dataset, a baidu drive link is https://pan.baidu.com/s/1bpJcLjx).

Some codes are borrowed from the [MASK RCNN IMPLEMENTATION](https://github.com/matterport/Mask_RCNN)

## Usage:
### Evaluation
'''
python LIP.py evaluate --model path_to_model.h5  --dataset  dataset_path/Single_Person --evalnum 0
'''
evalnum=0 uses the whole valset. A positive evalnum indicates the number of images to use for evaluation
 
### Test:

run 
```
demo.py 
```
to run test on some specific images (the main procedure is to call model.detect()) 

### Train
```
python LIP.py train --model path_to_model.h5  --dataset  dataset_path/Single_Person  trainmode pretrain
```
3 kinds of trainmodes available: pretrain, finetune, or fintune_ssloss_withdeep, which correspond to the 3 steps introduced in the paper Self-Supervised Neural Aggregation Networks for Human Parsing

Step1:
download [pspnet_pretrainweights](https://pan.baidu.com/s/1sloikGH)
run
```
python LIP.py train --model pspnet  --dataset  dataset_path/Single_Person  trainmode pretrain
```

set the parameters of model.train()  
```
epochs=40,layers='all'   
```

Step2 :
train the Neural Aggregation Networks
```
python LIP.py train --model pretain.h5(the best model generated in step1 )  --dataset  dataset_path/Single_Person  trainmode 
finetune
```

set the parameters of model.train() 
```
epochs=30,layers='head'  
```

Step3 :
train with Self-Supervised Loss
```
python LIP.py train --model finetune.h5(the best model generated in step2 )  --dataset  dataset_path/Single_Person  trainmode finetune_ssloss_withdeep
```

set the parameters of model.train()  
```
epochs=30,layers='psp5+'
```

The final Pretrain_model can be downloaded [here](https://pan.baidu.com/s/1nvMMl0P)


## Reference:
```
@inproceedings{Gong2017Look,
  title={Look into Person: Self-Supervised Structure-Sensitive Learning and a New Benchmark for Human Parsing},
  author={Gong, Ke and Liang, Xiaodan and Zhang, Dongyu and Shen, Xiaohui and Lin, Liang},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6757-6765},
  year={2017},
}

@inproceedings{Zhao2017Self,
  title={Self-Supervised Neural Aggregation Networks for Human Parsing},
  author={Zhao, Jian and Li, Jianshu and Nie, Xuecheng and Zhao, Fang and Chen, Yunpeng and Wang, Zhecan and Feng, Jiashi and Yan, Shuicheng},
  booktitle={Computer Vision and Pattern Recognition Workshops},
  pages={1595-1603},
  year={2017},
}
```
