# Forget to mitigate Robust Overfitting in Adversarial Training (FOMO)

This repository contains the official implementation of the ICLR 2024 paper **The effectiveness of random forgetting for robust generalization** [[Paper](https://openreview.net/forum?id=MEGQGNUfPx)] by **Vijaya Raghavan T Ramkumar, Elahe Arani and Bahram Zonooz** in [PyTorch](https://pytorch.org/). 

## Abstract
Deep neural networks are susceptible to adversarial attacks, which can compromise their performance and accuracy. Adversarial Training (AT) has emerged as a popular approach for protecting neural networks against such attacks. However, a key challenge of AT is robust overfitting, where the network's robust performance on test data deteriorates with further training, thus hindering generalization. Motivated by the concept of active forgetting in the brain, we introduce a novel learning paradigm called ``Forget to Mitigate Overfitting (FOMO)". FOMO alternates between the forgetting phase, which randomly forgets a subset of weights and regulates the model's information through weight reinitialization, and the relearning phase, which emphasizes learning generalizable features. Our experiments on benchmark datasets and adversarial attacks show that FOMO alleviates robust overfitting by significantly reducing the gap between the best and last robust test accuracy while improving the state-of-the-art robustness. Furthermore, FOMO provides a better trade-off between standard and robust accuracy, outperforming baseline adversarial methods. Finally, our framework is robust to AutoAttacks and increases generalization in many real-world scenarios.

![alt text](https://github.com/NeurAI-Lab/FOMO/blob/main/method_FOMO.png) 

For more details, please see the [Paper](https://openreview.net/forum?id=MEGQGNUfPx) and [Presentation](https://www.youtube.com/@neurai4080).

## Requirements

The code has been implemented and tested with `Python 3.8` and `PyTorch 1.12.1`.  To install the required packages: 
```bash
$ pip install -r requirements.txt
```


### Training 

Run [`LURE_main.py`](./LURE_main.py) for training the model in Anytiem framework with selective forgetting on CIFAR10 and CIFAR100. Run `ALMA.py` for training the model without selective forgetting which is the warm-started model. 

```
$ python .\LURE_main.py --data <data_dir> --log-dir <log_dir> --run <name_of_the_experiment> --dataset cifar10 --arch resnet18 \
--seed 10 --epochs 50 --decreasing_lr 20,40 --batch_size 64 --weight_decay 1e-4 --meta_batch_size 6250 --meta_batch_number 8 --snip_size 0.20 \
--save_dir <save-dir> --sparsity_level 1 -wb --gamma 0.1 --use_snip
```
For training the model with R-ImageNet, 

```
$ python ./LURE_main.py --data <data_dir> --imagenet_path <imagenet data path> --run <name_of_the_experiment> --dataset restricted_imagenet --arch resnet50 \
--seed 10 --epochs 50 --decreasing_lr 20,40 --batch_size 128 --weight_decay 1e-4 --meta_batch_size 6250 --meta_batch_number 8 --snip_size 0.20 \
--save_dir <save-dir> --sparsity_level 1 -wb --gamma 0.1 --use_snip

```
**Note Use `-buffer_replay`, `-no_replay` for training the model with buffer and without buffer data respectively. If no args is given then by default the model is trained in full replay setting.**




## Reference & Citing this work

If you use this code in your research, please cite the original works [[Paper](https://openreview.net/forum?id=MEGQGNUfPx)] :

```
@inproceedings{
ramkumar2024the,
title={The Effectiveness of Random Forgetting for Robust Generalization},
author={Vijaya Raghavan T Ramkumar and Bahram Zonooz and Elahe Arani},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=MEGQGNUfPx}
}

```

