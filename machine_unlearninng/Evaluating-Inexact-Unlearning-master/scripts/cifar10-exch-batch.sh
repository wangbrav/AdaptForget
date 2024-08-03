#!/bin/bash
#SBATCH -A $USER
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=conf80.txt
#SBATCH --mail-user=dheerajreddy.p@students.iiit.ac.in
#SBATCH --mail-type=ALL

#ARGS
chmod +x ../src/train.py
#logdir='./logs/oct_retina_qf100pvalue'
logdir='./logs/mlp_diabetes-qf50'
#dataset='diabetes'
dataset='Pima'
#dataset='retinamnist'
model='build_mlp_pima_model'
#num_classes=5
num_classes=2
confA=0
confB=1
num_change=25
#num_change=80
minlr_og=5e-3
maxlr_og=0.1
expname="Conf-C$confA-C$confB-$num_change[$minlr_og, $maxlr_og]_Batch64_62eps"

rf_L=9
rf_R=10
#rf_R=31
rf_S=1
#epochs_rf=10
epochs_rf=62
#epochs_ft=10
epochs_ft=62

#TRAIN
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)\
 python3 ../src/train.py --log-dir="$logdir" --exp-name="$expname"\
 --dataset="$dataset" --model="$model" --num-classes=$num_classes\
 --minlr-og="$minlr_og" --maxlr-og="$maxlr_og"\
 --confname="C$confA-C$confB-$num_change" --num-change=$num_change --exch-classes $confA $confB

#Filepath: must have path_oarg\npath_rarg\npath_o\npath_rpath_init
pathsout_tr="$logdir/$expname/train-paths.txt"
path_oarg=$(sed -n 1p "$pathsout_tr")
path_rarg=$(sed -n 2p "$pathsout_tr")
path_o=$(sed -n 3p "$pathsout_tr")
path_r=$(sed -n 4p "$pathsout_tr")
path_init=$(sed -n 5p "$pathsout_tr")

minlr_ft=$minlr_og
maxlr_ft=$maxlr_og
name_go='Golatkar'
name_rf="RetrFinal_L[$rf_L, $rf_R, $rf_S]_${epochs_rf}ep"
name_ft="Finetunefinal_LR[$minlr_ft, $maxlr_ft]_${epochs_ft}ep"

#UNLEARN
chmod +x ../src/unlearn.py
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)\
 python3 ../src/unlearn.py\
 --path-o="$path_o" --path-r="$path_r" --path-oarg="$path_oarg" --path-rarg="$path_rarg" --init-checkpoint="$path_init"\
 --num-classes=$num_classes\
 --golatkar=False --name-go="$name_go"\
 --retrfinal=True --name-rf="$name_rf" --epochs-rf=$epochs_rf --minL-rf=$rf_L --maxL-rf=$rf_R --stepL-rf=$rf_S\
 --finetune-final=True --name-ft="$name_ft" --epochs-ft=$epochs_ft --minlr-ft="$minlr_ft" --maxlr-ft="$maxlr_ft"\


# --finetune=True --name-ft="$name_ft" --epochs-ft=$epochs_ft --minlr-ft="$minlr_ft" --maxlr-ft="$maxlr_ft"\

#Filepath:  must have path_ntk\npath_fish\npath_ntkf\npath_ft\nprefix_rf
pathsout_un="$logdir/$expname/unlearn-paths.txt"
path_ntk=$(sed -n 1p "$pathsout_un")
path_fish=$(sed -n 2p "$pathsout_un")
path_ntkf=$(sed -n 3p "$pathsout_un")
path_ft=$(sed -n 4p "$pathsout_un")
prefix_rf=$(sed -n 5p "$pathsout_un")
# prefix_rf="$logdir/$expname/$name_rf/RetrFinal_"

chmod +x ../src/evaluation.py
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)\
 python3 ../src/evaluation.py\
 --path-o="$path_o"\
 --path-r="$path_r"\
 --path-oarg="$path_oarg"\
 --path-rarg="$path_rarg"\
 --num-classes=$num_classes --exch-classes $confA $confB\
 --golatkar=False --name-go="$name_go" --path-ntk="$path_ntk" --path-fisher="$path_fish" --path-ntkfisher="$path_ntkf"\
 --retrfinal=True --name-rf="$name_rf" --minL-rf=$rf_L --maxL-rf=$rf_R --stepL-rf=$rf_S --prefix-rf="$prefix_rf"\
 --finetune=True --name-ft="$name_ft" --path-ft="$path_ft"