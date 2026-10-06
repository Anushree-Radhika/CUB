import argparse
import importlib

if __name__ == '__main__':
    train_val = importlib.import_module('train-val')
    parser = argparse.ArgumentParser('TraitGen Training', parents=[train_val.get_args_parser()])
    args = parser.parse_args()
    train_val.main(args)
