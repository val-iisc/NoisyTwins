#!/bin/sh

TASK="evaluate" ## "train" "evaluate" "visualize"
DATASET="iNat19" ## "iNat19" "imagenet_lt" "CIFAR10_LT" "Imgnet_carniv" "AnimalFace_FS"

WANDB_API="" ## Fill your Personal WANDB Token Key Here.

if [ "${DATASET}" = "iNat19" ]; then
    # iNat19
    CFG="src/configs/iNat19/StyleGAN2-SPD-ADA-LC-NoisyTwins.yaml"
    DATA_DIR="./data"
    REF_SET="valid"
elif [ "${DATASET}" = "imagenet_lt" ]; then
    # ImageNet-LT
    CFG="src/configs/imagenet_lt/StyleGAN2-SPD-ADA-NoisyTwins.yaml"
    DATA_DIR="./data"
    REF_SET="valid"
elif [ "${DATASET}" = "CIFAR10_LT" ]; then
    # CIFAR10-LT
    CFG="src/configs/CIFAR10_LT/StyleGAN2-SPD-DiffAug-NoisyTwins.yaml"
    DATA_DIR="./data/cifar10"
    REF_SET="test"
elif [ "${DATASET}" = "Imgnet_carniv" ]; then
    # ImageNet-Carnivores
    CFG="src/configs/Imgnet_carniv/StyleGAN2-SPD-ADA-NoisyTwins.yaml"
    DATA_DIR="./data/ImageNet_Carnivores_20_100"
    REF_SET="train"
elif [ "${DATASET}" = "AnimalFace_FS" ]; then
    # AnimalFace-FS
    CFG="src/configs/AnimalFace_FS/StyleGAN2-SPD-ADA-NoisyTwins.yaml"
    DATA_DIR="./data/AnimalFace"
    REF_SET="train"
fi

if [ "${TASK}" = "train" ]; then
    
    METRICS="fid is" ## "fid is prdc"
    ##Evaluation during training by default run for Inception_V3 backbone 

    ## Fill the run name here to resume training from the checkpoint and uncomment -ckpt flag below.
    CKPT="" # "CIFAR10_LT-StyleGAN2-SPD-ADA-train-2022_07_29_12_14_51" 

    if [ "${DATASET}" = "iNat19" ] || [ "${DATASET}" = "imagenet_lt" ]; then 
        CUDA_VISIBLE_DEVICES=0,1 WANDB_API_KEY=$WANDB_API \
        python3 src/main.py  --data_dir $DATA_DIR -cfg $CFG \
        -ref $REF_SET -metrics $METRICS -l -hdf5 -t -v # -best -ckpt checkpoints/$CKPT

    elif [ "${DATASET}" = "CIFAR10_LT" ] || [ "${DATASET}" = "Imgnet_carniv" ] || [ "${DATASET}" = "AnimalFace_FS" ]; then
        CUDA_VISIBLE_DEVICES=0,1 WANDB_API_KEY=$WANDB_API \
        python3 src/main.py  --data_dir $DATA_DIR -cfg $CFG \
        -ref $REF_SET -metrics $METRICS -t -v # -best -ckpt checkpoints/$CKPT
    fi

elif [ "${TASK}" = "evaluate" ]; then
    ## Evaluate the trained models for various metrics

    EVAL_BACKBONE="Inception_V3" ## "CLIP", 
    ## CLIP is used only for evaluation of FID, and use only single GPU for evaluating using CLIP backbone (error otherwise). 

    METRICS="fid is prdc"  ## is prdc cannot be used with CLIP backbone.

    ## Fill run names in different lines in single string to evaluate multiple experiments one by one in loop.
    CKPTS=""  
    # "CIFAR10_LT-StyleGAN2-SPD-DiffAug-train-2022_10_20_15_43_42
    # CIFAR10_LT-StyleGAN2-SPD-DiffAug-train-2022_10_22_10_17_03"

    if [ "${DATASET}" = "iNat19" ] || [ "${DATASET}" = "imagenet_lt" ]; then 
        for CKPT in $CKPTS;do
            CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API WANDB_MODE=disabled \ 
            python3 src/main.py \
            --data_dir $DATA_DIR -cfg $CFG -l -hdf5 --eval_backbone $EVAL_BACKBONE \
            -metrics $METRICS -ref $REF_SET -v -ifid \
            -best -ckpt checkpoints/$CKPT 
        done
    elif [ "${DATASET}" = "CIFAR10_LT" ] || [ "${DATASET}" = "Imgnet_carniv" ] || [ "${DATASET}" = "AnimalFace_FS" ]; then
        for CKPT in $CKPTS;do
            CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API WANDB_MODE=disabled \ 
            python3 src/main.py \
            --data_dir $DATA_DIR -cfg $CFG --eval_backbone $EVAL_BACKBONE \
            -metrics $METRICS -ref $REF_SET -v -ifid \
            -best -ckpt checkpoints/$CKPT 
        done
    fi
    ## WANDB_MODE=disabled --> experiments are not logged to WANDB while evaluation
    ## -ifid --> evalautes the Intra class FID with corresponding backbone 

elif [ "${TASK}" = "visualize" ]; then

    METRICS="none"

    ## Fill run names in different lines in single string to evaluate multiple experiments one by one in loop.
    CKPTS=""  
    # "CIFAR10_LT-StyleGAN2-SPD-DiffAug-train-2022_10_20_15_43_42
    # CIFAR10_LT-StyleGAN2-SPD-DiffAug-train-2022_10_22_10_17_03"
    
    # ##Visualize and Analysis
    if [ "${DATASET}" = "iNat19" ] || [ "${DATASET}" = "imagenet_lt" ]; then 
        for CKPT in $CKPTS;do
            CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API WANDB_MODE=disabled \ 
            python3 src/main.py \
            --data_dir $DATA_DIR -cfg $CFG \
            -l -hdf5 -v -ref $REF_SET -best -ckpt checkpoints/$CKPT -metrics $METRICS 
        done
    elif [ "${DATASET}" = "CIFAR10_LT" ] || [ "${DATASET}" = "Imgnet_carniv" ] || [ "${DATASET}" = "AnimalFace_FS" ]; then
        for CKPT in $CKPTS;do
            CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=$WANDB_API WANDB_MODE=disabled \ 
            python3 src/main.py \
            --data_dir $DATA_DIR -cfg $CFG \
            -v -ref $REF_SET -best -ckpt checkpoints/$CKPT -metrics $METRICS 
        done
    fi
fi