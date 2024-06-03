import torch.cuda
import torch.nn.functional as F
import sys
import time
from texttable import Texttable
from loguru import logger as lg
from param_parser import get_parser

args = get_parser()


def logger(info):
    epoch, t, best_epoch = info['epoch'], info['time'], info['best_epoch']
    train_acc, test_acc, val_acc, train_loss = info['train_acc'], info['max_test_acc'], info['max_val_acc'], info[
        'train_loss']
    # print(
    #     'epoch: {:d}, best_epoch: {:d}, Train Acc: {:.3f}, Test Accuracy: {:.3f}, Train Loss: {:.3f}, Time: {:.3f} s'.format(
    #         epoch, best_epoch, train_acc, test_acc, train_loss, (time.time() - t)))
    lg.info(
        'epoch: {:d}, best_epoch: {:d}, Train Acc: {:.3f}, Test Accuracy: {:.3f}, Val Accuracy: {:.3f}, Train Loss: {:.3f}, Time: {:.3f} s'.format(
            epoch, best_epoch, train_acc, test_acc, val_acc, train_loss, (time.time() - t)))
    # lg.info()

    sys.stdout.flush()


def cuda_device(x):
    if len(args.devices) > 1:
        out = x.cuda()
    else:
        out = x.cuda(args.devices[0])
    return out


def stop():
    print(0 / 0)


def sizee(a):
    print("size is:", a.size())
