---
tags: thesis, experiment
---

# [Advancing Multi-Criteria Chinese Word Segmentation Through Criterion Classification and Denoising](https://aclanthology.org/2023.acl-long.356/) (ACL 2023)

# Get Start

## Download datasets
The datasets need to be downloaded from [Sighan 2005](http://sighan.cs.uchicago.edu/bakeoff2005). 
```
./download_data.sh 
```

Other datasets we used in our paper can be downloaded from [hankcs/multi-criteria-cws](https://github.com/hankcs/multi-criteria-cws/tree/master/data/other).

## Install packages
All the packages used in this project is listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Preprocess

Our preprocess includes change a sequence of numbers to "0" and a sequence of alphabets to "a" while we are training. We also need to do these preprocess for the test set. These word will be recovered by our postprocess.

```shell
python3 -m MCCWS.script.preprocess \
    --original_gold_path ./icwb2-data/gold/as_testing_gold.utf8 \
    --new_path ./data/testset/as_test.txt
```
`--original_gold_path` : The path of the original test set.  

`--new_path` : The directory where you want to store the preprocessed test set. 

## Train valid split

Split the original training dataset into two parts. 90% of the data will be used for training and the rest will be used as validation set.

```shell
python3 -m MCCWS.script.train_valid_split \
    --split_dataset ./data/trainset/as_train.txt \
    --new_train_file_path ./data/trainset/as_train.txt \
    --new_valid_file_path ./data/trainset/as_valid.txt
```

`--split_dataset` : The path of the original training set.  

`--new_train_file_path` : The directory where you want to store the new training set. 

`--new_valid_file_path` : The directory where you want to store the new validation set.
# Start finetune

## Script Explianation
We can start finetune our MCCWS model by the script below.
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m MCCWS.script.finetune \
    --batch_size 16 \
    --dropout 0.1 \
    --lr 2e-5 \
    --smooth_factor 0.1 \
    --replace_rate $REPLACE_RATE \
    --epoch 10 \
    --seed $SEED \
    --model_name bert-base-chinese \
    --exp_name testing \
    --trainset_token "[ALL]" \
    --accumulation_step 4 \
    --start_log 280000 \
    --save_step 1000 \
    --gpu 0
```
`--batch_size` : Set the batch size use for training.  ( `default=16`)

`--dropout` : Set the dropout rate use training. ( `default=0.1`) 

`--lr` : Learning rate used for training. (`default=2e-5`) 

`--smooth_factor` : The value of label smoothing used for training Chinese word segmentation task. (`default=0.1`) 

`--replace_rate` : The replace rate of criterion token. If we set the value to 0.1, then one-tenth of the data in the dataset will have a [UNC] token as their criteria token. 

`--seed` : Random seed that used for training.

`--model_name` : Select the model from [Hugging Face](https://huggingface.co/). (`default : bert-base-chinese`)

`--exp_name` : The name of your experiment. 

`--trainset_token` : Select the dataset used for training. More details can see in `./MCCWS/util/_dataset.py`. (`default : "[ALL]"`) 

`--accumulation_step` : The step of gradient accumulation. (`default : 4`) 

`--start_log` : The step that starts to store the model. (`default : 280000`) 

`--save_step` : Model saving period. (`default : 1000`)

`--gpu` : Select the GPU used for training. (`default : 0`)


Use `./finetune.sh` can also do the same thing.

# Evaluation

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m MCCWS.inference.infer_all \
        --model_name bert-base-chinese \
        --exp_name $EXP_NAME \
        --criteria_token "[AS]" \
        --testset_token "[AS]" \
        --criteria 10 \
        --start_checkpoint $START \
        --end_checkpoint $END \
        --checkpoint_step $STEP \
        --gpu $GPU
```
`--model_name` : Select the model from [Hugging Face](https://huggingface.co/). (`default : bert-base-chinese`)


`--exp_name` : The name of your experiment. 

`--criteria_token` : The criteria token you want to use.

`--testset_token` : Specify the testing set you want to test.

`--criteria` : How many criteria did you used in training set. (`default : 10`)

`--start_checkpoint` : The first checkpoint of the model that you want to use for testing. 

`--end_checkpoint` : The last checkpoint of the model that you want to use for testing. 

`--checkpoint_step` : Model testing period.

`--gpu` : Select the GPU used for training. (`default : 0`)

# Other Information

If you have other questions, please feel free to open some issue.

```
@inproceedings{chou-etal-2023-advancing,
    title = "Advancing Multi-Criteria {C}hinese Word Segmentation Through Criterion Classification and Denoising",
    author = "Chou, Tzu Hsuan  and
      Lin, Chun-Yi  and
      Kao, Hung-Yu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.356",
    doi = "10.18653/v1/2023.acl-long.356",
    pages = "6460--6476",
    abstract = "Recent research on multi-criteria Chinese word segmentation (MCCWS) mainly focuses on building complex private structures, adding more handcrafted features, or introducing complex optimization processes.In this work, we show that through a simple yet elegant input-hint-based MCCWS model, we can achieve state-of-the-art (SoTA) performances on several datasets simultaneously.We further propose a novel criterion-denoising objective that hurts slightly on F1 score but achieves SoTA recall on out-of-vocabulary words.Our result establishes a simple yet strong baseline for future MCCWS research.Source code is available at https://github.com/IKMLab/MCCWS.",
}
```