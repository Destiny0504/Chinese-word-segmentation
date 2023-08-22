import re

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class split_dataset:
    def __init__(self, datasets) -> None:

        self.bigram = {}

        for dataset_token, dataset_paths in datasets.items():
            for dataset_path in dataset_paths:
                with open(f"{dataset_path}", "r", encoding="utf8") as f:
                    content = f.readlines()
                self.data = self.fullwidth2halfwidth(content)
                self.train, self.valid = self.split_data()

    def fullwidth2halfwidth(self, data):
        next_line = re.compile(r"\n")
        tmp = []
        for string in data:
            string = next_line.sub("", string)
            sentence = ""
            for character in string:
                character = ord(character)
                if character == 12288:
                    character = 32
                elif 65281 <= character <= 65374:
                    character -= 65248
                sentence += chr(character)
            tmp.append(sentence)
        return tmp

    def split_data(self):
        """Split the dataset into training and testing set.
           The training set would have nine-tenth of data in the original dataset.

        Returns:
            _type_: _description_
        """
        return train_test_split(
            self.data, test_size=0.1, random_state=42, shuffle=False
        )

    def bigram_calculation(self):
        space = re.compile(r"\s")
        for sentence in tqdm(self.data):
            i = 0
            while i < len(sentence) - 2:
                i += 1
                if sentence[i + 1] != " ":
                    try:
                        self.bigram[sentence[i : i + 2]][0] += 1
                    except:
                        self.bigram[sentence[i : i + 2]] = [0, 0]
                        self.bigram[sentence[i : i + 2]][0] += 1
                else:
                    try:
                        self.bigram[space.sub("", sentence[i : i + 3])][1] += 1
                    except:
                        self.bigram[space.sub("", sentence[i : i + 3])] = [0, 0]
                        self.bigram[space.sub("", sentence[i : i + 3])][1] += 1
                    i += 1

        return self.bigram


if __name__ == "__main__":
    datasets = {"[AS]": ["./data/trainset/as_training.utf8"]}
    a = split_dataset(datasets=datasets)
    b = split_dataset(datasets={"[CIT]": ["./data/trainset/cityu_training.utf8"]})
    b.bigram_calculation()
    # print(a.valid)
    for k, v in a.bigram_calculation().items():
        if v[0] > 20 or v[1] > 20:
            if (
                b.bigram.get(k) is not None
                and (
                    torch.Tensor(b.bigram.get(k)).softmax(dim=-1)[0]
                    - torch.Tensor(a.bigram.get(k)).softmax(dim=-1)[0]
                )
                > 0.2
            ):
                print(f"bigram : {k} fraction : {b.bigram.get(k)}")
                print(f"bigram : {k} fraction : {v}")
    # print(f'bigram : {"就是"} fraction : {a.bigram["就是"]}')
    # print(a.bigram_calculation())
