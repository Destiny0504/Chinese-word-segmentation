import re

from tqdm import tqdm


class preprocess:
    def __init__(self, datasets) -> None:

        for dataset_token, dataset_paths in datasets.items():
            for dataset_path in dataset_paths:
                with open(f"{dataset_path}", "r", encoding="utf8") as f:
                    content = f.readlines()
                self.original_data = self.fullwidth2halfwidth(content)
                data = self.replace_num(self.original_data)
                self.data = self.replace_eng(data)
                # self.data = self.replace_strange_punctuation(data)
                # self.data = self.replace_num_eng_combination(data)

    def fullwidth2halfwidth(self, data):
        """Convert full width character to half width character

        Args:
            data (List[str]): Sentences which needs to convert to half width.

        Returns:
            converted_str (List[str]): Converted sentences.
        """
        next_line = re.compile(r"\n")
        converted_str = list()
        for string in data:
            string = next_line.sub("", string)
            sentence = ""
            for character in string:
                character = ord(character)

                # Turn full size space into half size space
                if character == 12288:
                    character = 32

                # Turn full size character into half size character
                elif 65281 <= character <= 65374:
                    character -= 65248
                sentence += chr(character)
            converted_str.append(sentence)
        return converted_str

    def replace_num(self, data):
        """Replace the numbers in the dataset with a character '0'.

        Args:
            data (List[str]): Sentences which need to be preprocessed.

        Returns:
            data (List[str]): Converted sentences.
        """
        numbers = re.compile(r"((-|\+)?(\d+)([\.|·/∶:]\d+)?%?)+")
        data = list(map(lambda x: numbers.sub("0", x), data))
        return data

    def replace_eng(self, data):
        """Replace the alphabets in the dataset with a character 'a'.

        Args:
            data (List[str]): Sentences which need to be preprocessed.

        Returns:
            data (List[str]): Converted sentences.
        """
        english_char = re.compile(r"[A-Za-z_.]+")
        data = list(map(lambda x: english_char.sub("a", x), data))
        return data

    def replace_num_eng_combination(self, data):
        """Replace a word combined with number and alphabet in the dataset with a character 'c'.
            (not used)

        Args:
            data (List[str]): Sentences which need to be preprocessed.

        Returns:
            data (List[str]): Converted sentences.
        """
        combination = re.compile(r"(a|0){2,}")
        data = list(map(lambda x: combination.sub("c", x), data))
        return data

    def recover(self):
        recovered_data = []
        for idx, data in enumerate(self.data):
            tmp = []
            hold = ""
            data = data.split()
            original_data = self.original_data[idx].split()
            for counter, word in enumerate(data):
                if "a" in word or "0" in word or "c" in word:
                    tmp.append(hold)
                    tmp.append(original_data[counter])
                    hold = ""
                else:
                    hold += word
            if hold != "":
                tmp.append(hold)
            recovered_data.append(" ".join(tmp))
        return recovered_data

    def test_data(self):
        # a function to test the your converted data
        test_data = []
        punctuation = re.compile(r"^[。！？：；…+、，（）“”’,;!?、,]+$")
        for idx, data in enumerate(self.data):
            tmp = []
            hold = ""
            data = data.split()
            for counter, word in enumerate(data):
                if len(punctuation.sub("", word)) == 0:
                    tmp.append(hold)
                    tmp.append(word)
                    hold = ""
                else:
                    hold += word
            if hold != "":
                tmp.append(hold)
            test_data.append(" ".join(tmp))
        return test_data
