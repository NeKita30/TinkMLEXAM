import argparse
import os
import re
import numpy as np
import pickle


class NModel:
    def __init__(self, in_dir):
        list_of_files = os.listdir(in_dir)
        self.list_of_text = [self.__prepare_file(os.path.join(in_dir, text_file))
                             for text_file in list_of_files]
        self.number_of_word = dict()
        self.contin_for_mo_prefix = dict()
        self.contin_for_bi_prefix = dict()
        self.contin_for_three_prefix = dict()
        self.counter = 0

    def fit(self):
        for text in self.list_of_text:
            for i in range(len(text)):
                word = text[i]
                if i < len(text) - 1 and word not in {'.', ','}:
                    self.number_of_word[word] = \
                        self.number_of_word.setdefault(word, 0) + 1
                    self.counter += 1

                if i > 0:
                    first_word = text[i - 1]
                    self.contin_for_mo_prefix[first_word][word] = \
                        self.contin_for_mo_prefix.setdefault(first_word, dict()).setdefault(word, 0) + 1

                if i > 1:
                    second_word = text[i - 1]
                    first_word = text[i - 2]
                    self.contin_for_bi_prefix[(first_word, second_word)][word] = \
                        self.contin_for_bi_prefix.setdefault((first_word, second_word), dict()).setdefault(word, 0) + 1

                if i > 2:
                    third_word = text[i - 1]
                    second_word = text[i - 2]
                    first_word = text[i - 3]
                    self.contin_for_three_prefix[(first_word, second_word, third_word)][word] = \
                        self.contin_for_three_prefix.setdefault((first_word, second_word, third_word),
                                                                dict()).setdefault(word, 0) + 1

    def generate(self, length, prefix=[]):
        gen_text = prefix
        while len(gen_text) < length:
            first = gen_text[-3] if len(gen_text) > 2 else "__"
            second = gen_text[-2] if len(gen_text) > 1 else "__"
            third = gen_text[-1] if len(gen_text) > 0 else "__"
            if (first, second, third) in self.contin_for_three_prefix:
                if (len(self.contin_for_three_prefix[(first,
                                                      second,
                                                      third)]) > len(self.contin_for_bi_prefix[(second,
                                                                                                third)])) or \
                        ({first, second, third} & {'.', ','}):
                    word_number = np.array(list(self.contin_for_three_prefix[(first, second, third)].items()))
                else:
                    word_number = np.array(list(self.contin_for_bi_prefix[(second, third)].items()))
            elif (second, third) in self.contin_for_bi_prefix:
                word_number = np.array(list(self.contin_for_bi_prefix[(second, third)].items()))
            elif third in self.contin_for_mo_prefix:
                word_number = np.array(list(self.contin_for_mo_prefix[third].items()))
            else:
                word_number = np.array(list(self.number_of_word.items()))
            words = word_number[:, 0]
            numbers = np.array(word_number[:, 1], dtype=float)
            gen_text = np.concatenate((gen_text,
                                       np.random.choice(words, 1,
                                                        p=numbers / np.array([numbers.sum()] * len(word_number)))))
        return ' '.join(gen_text)

    @staticmethod
    def __prepare_file(text_file):
        with open(text_file, 'r') as f:
            file_content = f.read()
            file_list_of_words = re.split(r'\s', file_content)

            file_text = []
            for word in file_list_of_words:
                word_clear = re.sub(r'[\W_]+', '', word).lower()
                if word_clear:
                    file_text.append(word_clear)
                    if word[-1] == '.' or word[-1] == ',':
                        file_text.append(word[-1])

            return file_text

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", default="", type=str, dest="input")
    parser.add_argument("--model", required=True, type=str)
    args = parser.parse_args()

    input_dir = args.input
    output_file = args.model

    mdl = NModel(input_dir)
    mdl.fit()
    with open(output_file, 'wb') as file:
        pickle.dump(mdl, file)
