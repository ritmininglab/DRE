# DRE
This this the code for the paper "Distributionally Robust Ensemble of Lottery Tickets Towards Calibrated Sparse Network Training" which is accepted by NeurIPS2023"


To train each base learner run the following command 

python3 main.py --config config_name   --multigpu 0 --data data_path --arch  $arch  --prune-rate $prune_rate --seed $run  --lamb $lamb

Parameters:
For Cifar10: config_name -> configs/smallscale/conv8/conv8_usc_unsigned.yml, arch -> choose one of the following [cResNet50, cResNet101]
For Cifar100: config_name -> configs/cifar100/subnetonly/hnn.yml, arch -> choose one of the following [c100ResNet101, c100ResNet152]
For TinyImageNet: config_name -> configs/tinyimagenet/subnetonly/hnn.yml, arch -> choose one of the following [tResNet101, WideResNet101_2]
lamb -> Choose value between 0 to \infty higher the value, more emphasis on all easy samples. lamb = 0 is same as that of considering most difficult 
        and \infty is the one where we give equal emphasis to all samples (ERM)
        For Cifar10 base models with lamb = [10, 500, \infty]
        For Cifar100 base models with lamb = [50, 500, \infty]
        For TinyImageNet base models with lamb = [100, 1000000, \infty]
prune_rate -> percentage of NN weights we want to keep. Options: [3%, 5%]
data -> Path of dataset should have Dataset. Example /home/ak23/Dataset
seed -> Run number (model) number 

Testing:
Once we train each model, we can perform inference on testing dataset which will save outputs (logit values), gts in the outputs folder. This can be done by following 
python3 main.py --config config_name   --multigpu 0 --data data_path --arch  $arch  --prune-rate $prune_rate --seed $run  --lamb $lamb --evaluate

Once we get output, we ensemble them in logit layer by taking mean of output for each class. Then, we take softmax and
measure accuracy and ECE. This can be done by running the following command 

python3 test_performance_comparison_cifar10.py (for Cifar10)


Note: Base model with lamb = \infty is same as without passing lamb i.e., to train the base model with amb = \infty, simply perform following 

python3 main.py --config config_name   --multigpu 0 --data data_path --arch  $arch  --prune-rate $prune_rate --seed $run

For inference perform following 
python3 main.py --config config_name   --multigpu 0 --data data_path --arch  $arch  --prune-rate $prune_rate --seed $run --evaluate
