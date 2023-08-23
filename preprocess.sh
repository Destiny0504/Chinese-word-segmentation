# Our preprocess includes change a sequence of numbers to "0" and a sequence of alphabets to "a" while we are training.
# So we also need to do these preprocess for the test set.

# original_gold_path is the path of the test/train gold file.
# new_path is the path were you want to store the preprocessed file.

python3 -m MCCWS.script.preprocess \
    --original_gold_path ./icwb2-data/gold/as_testing_gold.utf8 \
    --new_path ./data/testset/as_test.txt

python3 -m MCCWS.script.preprocess \
    --original_gold_path ./icwb2-data/training/as_training.utf8 \
    --new_path ./data/trainset/as_train.txt

python3 -m MCCWS.script.train_valid_split \
    --split_dataset ./data/trainset/as_train.txt \
    --new_train_file_path ./data/trainset/as_train.txt \
    --new_valid_file_path ./data/trainset/as_valid.txt
