import argparse
import re
import pickle
import time
from train import NModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--length", required=True, type=int)
    args = parser.parse_args()

    with open(args.model, 'rb') as file:
        mdl = pickle.load(file)
        mdl.set_seed(int(time.time()))

        prefix = [re.sub(r'[\W_]+', '', word).lower()
                  for word in re.split(r'\s', args.prefix)]
        if prefix == [""]:
            prefix = []
        text = mdl.generate(args.length, prefix)
        print(text)
