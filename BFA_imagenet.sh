#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/fpo/anaconda3/envs/ba/bin/python" # python environment path
    TENSORBOARD='/home/fpo/anaconda3/envs/ba/bin/tensorboard' # tensorboard environment path
#    data_path='/home/fpo/data/cifar10' # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/
    cd ./save 
    mkdir ./${DATE}/
    cd ..
fi

############### Configurations ########################
enable_tb_display= false # enable tensorboard display
#model=resnet34_quan 
#model=mobilenet_v2_quan 
#model=resnet18_quan
model=alexnet_quan
dataset=cifar100
#dataset=svhn
#dataset=stl10
data_path='/home/fpo/data/cifar100' # dataset path
#test_batch_size=256
test_batch_size=128
#resume_path='/home/fpo/Bitflip/save/2021-06-16/cifar10_alexnet_quan/checkpoint.pth.tar'
resume_path='/home/fpo/Bitflip/save/{2021-06-21}/cifar100_alexnet_quan/model_best.pth.tar'
save_path=/home/fpo/Bitflip/save/${DATE}/${dataset}_${model}

attack_sample_size=128 # number of data used for BFA
# n_iter=20 # number of iteration to perform BFA
n_iter=5
k_top=10 # only check k_top weights with top gradient ranking in each layer
#tb_path=  /home/fpo/Bitflip/save/BFA/${dataset}_${model}/tb_log  #tensorboard log path

############### Neural network ############################
{
#for ((i=1;i<=30;i++));
#do
 python main.py --dataset ${dataset}   --data_path ${data_path}     --arch ${model} --save_path ${save_path}      --test_batch_size ${test_batch_size} --workers 8 --ngpu 1 --gpu_id 1     --print_freq 50      --n_iter ${n_iter} --k_top ${k_top} --attack_sample_size ${attack_sample_size} --resume ${resume_path} --bfa 
#python main.py --dataset ${dataset}   --data_path ${data_path}     --arch ${model} --save_path ${save_path}      --test_batch_size ${test_batch_size} --workers 8 --ngpu 1 --gpu_id 1    --print_freq 50   --k_top ${k_top} 
#python main.py --dataset ${dataset}   --data_path ${data_path}     --arch ${model} --save_path ${save_path}      --test_batch_size ${test_batch_size} --workers 8 --ngpu 0  --gpu_id 0    --print_freq 50   --k_top ${k_top} 
#done
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 1 
    wait
    echo "tensorboard exe"
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 2
    wait
    echo "display"
    case $HOST in
    "Hydrogen")
	 echo "firefox"
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
	echo "google"
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait
