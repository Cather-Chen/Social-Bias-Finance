# coding: UTF-8
import time
import torch
import numpy as np
from utils import build_dataset, build_iterator, get_time_dif
from train_eval import train, init_network, test
from importlib import import_module
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Multi-label classifier')
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':

    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    if torch.cuda.is_available():
        print("training on gpu")
    else:
        print("training on cpu")
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Data-finished Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    #test(config, model, test_iter)
    train(config, model, train_iter, dev_iter, test_iter)
