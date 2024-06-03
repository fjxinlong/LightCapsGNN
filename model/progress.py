import sys
import time
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam, SGD
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DenseDataLoader as DenseLoader
import numpy as np

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from loguru import logger as lg
from model.capsgnn import CapsGNN
from model.loss import margin_loss
from utils import logger, stop, cuda_device


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def k_folds_cross_validation(args, dataset, max_node_num, folds):
    epochs, batch_size, lr, lr_decay_factor = args.epochs, args.batch_size, args.lr, args.lr_decay_factor
    lr_decay_step_size, weight_decay, epoch_select = args.lr_decay_step_size, args.weight_decay, args.epoch_select
    with_eval_mode = args.with_eval_mode

    output_device = args.devices[0] if type(
        args.devices) is list else args.devices

    assert epoch_select in ['val_min', 'test_max'], epoch_select
    val_losses, train_accs, test_accs, val_accs = [], [], [], []
    test_acc_tmp = 0
    sign = 0
    test_accuracies1 = []
    val_accuracies1 = []
    test_accuracies2 = []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds, epoch_select))):
        # print(fold)
        # if fold < 4:
        #     continue
        set_seed(args.seed)
        print(f"------------fold : {fold + 1}---------------")
        train_dataset, test_dataset, val_dataset = dataset[train_idx], dataset[test_idx], dataset[val_idx]
        train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)

        model = cuda_device(CapsGNN(args, max_node_num, dataset.num_features, dataset.num_classes))

        # model = CapsGNN(args, dataset.num_features, dataset.num_classes).to(output_device)
        if type(args.devices) is list:
            if len(args.devices) > 1:
                model = nn.DataParallel(
                    model,
                    device_ids=args.devices,
                )
        total = sum([param.nelement() for param in model.parameters()])

        print("Number of parameter: %.2fK" % (total/1e3))
        # 0/0

        # print(model.parameters())
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        writer = SummaryWriter(log_dir=args.log_path + '/Graph/' + current_time)

        patience_cnt = 0
        min_loss = 1e10

        bar = tqdm(range(1, epochs + 1), ncols=150)
        t_fold = time.time()
        best_epoch = 0

        max_val_acc = 0
        min_val_loss = 100000
        max_test_acc1 = 0
        max_test_acc2 = 0
        max_eopch = 0

        for epoch in bar:
            train_loss, train_acc = train(
                args, model, optimizer, train_loader, output_device, max_node_num, epoch, writer)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            train_accs.append(train_acc)
            val_loss, val_acc = evaluation(args, model, val_loader, output_device, with_eval_mode)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            test_loss, test_acc = evaluation(args, model, test_loader, output_device, with_eval_mode)
            test_accs.append(test_acc)
            writer.add_scalar('test_acc', test_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            # print("----------",model.first_capsule.gcn.state_dict())
            if max_val_acc < val_acc:
                max_val_acc = val_acc
                max_test_acc1 = test_acc
                max_eopch = epoch
            if max_val_acc == val_acc:
                max_val_acc = val_acc
                if max_test_acc1 < test_acc:
                    max_test_acc1 = test_acc
                    max_eopch = epoch
                #
                # max_test_acc1 = max(test_acc, max_test_acc1)
                # if max_test_acc1 == test_acc:
                #     max_eopch = epoch

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                max_test_acc2 = test_acc
                # max_eopch = epoch
            if min_val_loss == val_loss:
                min_val_loss = val_loss
                max_test_acc2 = max(test_acc, max_test_acc2)
                # if max_test_acc2 == test_acc:
                #     max_eopch = epoch

            if val_losses[-1] < min_loss:
                min_loss = val_losses[-1]
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1

            bar.set_description(f"epoch:{epoch}")
            bar.set_postfix(loss=train_loss, train_acc=train_accs[-1], test_acc=test_accs[-1], val_acc=val_accs[-1])
            # if logger is not None:
            #     logger(eval_info)

            if patience_cnt == args.patience:
                break
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            # print("test_acc::", test_acc)
        test_accuracies1.append(max_test_acc1)
        test_accuracies2.append(max_test_acc2)
        # train_acc, test_acc = tensor(train_accs), tensor(test_accs)
        # train_acc = train_acc.view(fold + 1, epochs)
        # test_acc = test_acc.view(fold + 1, epochs)
        eval_info = {
            'epoch': epochs,
            'train_loss': train_loss,
            'train_acc': train_accs[-1],
            'max_test_acc': max_test_acc1,
            'max_val_acc': max_val_acc,
            'time': t_fold,
            'best_epoch': max_eopch
        }
        if logger is not None:
            logger(eval_info)
        # break
        # return time.time()-t_fold, total/1e3
        if max_test_acc1 < args.threshold:
            sign = 1
            break
        # break
        return max_test_acc1, max_test_acc2
    # if sign == 1:
    #     return 0, 0

    # avg_test1 = tensor(test_accuracies1)
    # test_acc_std1 = avg_test1.std().item()
    # test_acc_mean1 = avg_test1.mean().item()
    # avg_test2 = tensor(test_accuracies2)
    # test_acc_std2 = avg_test2.std().item()
    # test_acc_mean2 = avg_test2.mean().item()

    # lg.info(('final result ACC_max: Test Acc: {:.2f} {:.2f},Total Time: {:.3f}s'.format(test_acc_mean1 * 100,
    #                                                                                      test_acc_std1 * 100,
    #                                                                                      (time.time() - t_all))))

    # lg.info(('final result Loss_min: Test Acc: {:.2f} {:.2f},Total Time: {:.3f}s'.format(test_acc_mean2 * 100,
    #                                                                                     test_acc_std2 * 100,
    #                                                                                     (time.time() - t_all))))
    # sys.stdout.flush()
    # return test_acc_mean1, test_acc_mean2
    # return test_acc_mean, test_acc_std


def k_fold(dataset, folds, epoch_select):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        # in this situation, val_invices = test_indices
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))
    return train_indices, test_indices, val_indices


def train(args, model, optimizer, loader, device, max_node_num, epoch, writer):
    model.train()
    total_loss = 0
    correct = 0
    for batch_iter, data in enumerate(loader):
        optimizer.zero_grad()
        data_batch = data.to(device)
        features_batch, adj_batch, mask_batch = data.x, data.adj, data.mask
        # out, _ = model(features_batch, adj_batch, data.y, mask_batch)
        out, reconstruction_loss = model(features_batch, adj_batch, data.y, mask_batch)
        # out, reconstruction_loss, l1_loss = model(features_batch, adj_batch, data.y, mask_batch)

        # writer.add_graph(model,(features_batch, adj_batch, data.y, mask_batch))
        loss_ce = margin_loss(out, data_batch.y.view(-1))
        reconstruction_loss = reconstruction_loss.mean()

        # loss = loss_ce + args.theta * reconstruction_loss+0.1*l1_loss
        loss = loss_ce + args.theta * reconstruction_loss
        # loss = loss_ce


        pred = out.max(1)[1]
        correct += pred.eq(data_batch.y.view(-1)).sum().item()
        # loss = loss.requires_grad_()
        loss.backward()
        total_loss += loss.item() * data_batch.x.size(0)
        writer.add_scalar('CE loss', loss_ce, (epoch - 1) * len(loader) + (batch_iter + 1))
        writer.add_scalar('Reconstruction loss', args.theta *
                          reconstruction_loss, (epoch - 1) * len(loader) + (batch_iter + 1))
        optimizer.step()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluation(args, model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    loss = 0
    correct = 0
    for data in loader:
        data_batch = data.to(device)
        with torch.no_grad():
            features_batch, adj_batch, mask_batch = data.x, data.adj, data.mask
            out, reconstruction_loss = model(features_batch, adj_batch, data.y, mask_batch)
            # out, reconstruction_loss, l1_loss = model(features_batch, adj_batch, data.y, mask_batch)
            pred = out.max(1)[1]
        correct += pred.eq(data_batch.y.view(-1)).sum().item()
        # loss += (margin_loss(out, data_batch.y.view(-1)).item())
        loss += (margin_loss(out, data_batch.y.view(-1)).item() +
                 args.theta * reconstruction_loss.item())
        # loss += (margin_loss(out, data_batch.y.view(-1)).item() +
        #          args.theta * reconstruction_loss.item()+0.1*l1_loss.item())

    return loss / len(loader.dataset), correct / len(loader.dataset)
