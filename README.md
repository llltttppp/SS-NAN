# SS-NAN
keras implement of the paper Self-Supervised Neural Aggregation Networks for Human Parsing

This code implement three kinds model for LIP dataset(Look into Person: Self-supervised Structure-sensitive Learning and A New
Benchmark for Human Parsing)

Data preparing:please download the LIP(single person train,val dataset, a baidu drive link is https://pan.baidu.com/s/1bpJcLjx ) and place a right place

Some codes are borrowed from https://github.com/matterport/Mask_RCNN

Usage:
Evaluation

python LIP.py evaluate --model path_to_model.h5  --dataset  dataset_path/Single_Person --evalnum 0 (0 for all valset,other number indicate the number of images use for evaluation)

Test:
run demo.py to run test on some specific images (main procedure is call model.detect() function) 

Train
for training
python LIP.py train --model path_to_model.h5  --dataset  dataset_path/Single_Person  trainmode pretrain/finetune/fintune_ssloss_withdeep

three trainmode correspond to three step claimed in the paper Self-Supervised Neural Aggregation Networks for Human Parsing
Step1:
download pspnet_pretrainweights to the dir(link is https://pan.baidu.com/s/1sloikGH)
run
python LIP.py train --model pspnet  --dataset  dataset_path/Single_Person  trainmode pretrain
set the parameters of model.train()  epochs=40,layers='all'      
Step2 :
train the Neural Aggregation Networks
python LIP.py train --model pretain.h5(the best model generated in step1 )  --dataset  dataset_path/Single_Person  trainmode 
finetune

set the parameters of model.train()  epochs=30,layers='head'   

Step3 :
train with Self-Supervised 
python LIP.py train --model finetune.h5(the best model generated in step2 )  --dataset  dataset_path/Single_Person  trainmode finetune_ssloss_withdeep

set the parameters of model.train()  epochs=30,layers='psp5+'


Final Pretrain_model can find here(https://pan.baidu.com/s/1nvMMl0P)

results:
Pixel Accuracy:85.8   MeanAccuracy:58.1    MeanIOU:47.90



Reference:

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
