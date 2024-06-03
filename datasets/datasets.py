import os.path as osp
import re
import shutil
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch
import csv
from collections import Counter

from datasets.feature_expansion import FeatureExpander


def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None):
    if root is None or root == '':
        path = osp.join(osp.expanduser('~'), 'pyG_data', name)
    else:
        path = osp.join(root)
        if osp.exists(osp.join(path, name)) is False:
            dataset = TUDataset(path, name)
    degrees = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0
    coord = feat_str.find("coord") >= 0

    dir_name = osp.join(path, name)
    if osp.exists(osp.join(dir_name, 'processed')) is True:
        shutil.rmtree(osp.join(dir_name, 'processed'))
    # Obtain max_node_num
    with open(osp.join(path, name, "raw", name + "_graph_indicator.txt")) as f:
        temp = list(csv.reader(f))
    out = [int(i[0]) for i in temp]
    # count=0
    # for i in range(len(Counter(out).most_common())):
    #     count+=Counter(out).most_common()[i][1]
    # # print(Counter(out).most_common()[4])
    # print(count/len(Counter(out).most_common()))

    max_node_num = Counter(out).most_common(1)[0][1]
    # only maintain the 1000 nodes with the maximum degree
    if max_node_num > 1000:
        max_node_num = 1000
    with open(osp.join(path, name, "raw", name + "_A.txt")) as f:
        temp = list(csv.reader(f))
    out = [(int(i[0]), int(i[1])) for i in temp]
    out = torch.tensor(out).transpose(0, 1)

    row, col = out
    num_nodes = row.max().item()


    deg = degree(row - 1, num_nodes)
    max_deg = deg.max()

    if max_deg < 10:
        onehot_maxdeg = 10
    elif max_deg < 50:
        onehot_maxdeg = 50
    pre_transform = FeatureExpander(max_node_num=max_node_num,
        degree=degrees, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        group_degree=groupd).transform

    dataset = TUDataset(
        path, name, pre_transform=pre_transform,
        use_node_attr=True)

    dataset.data.edge_attr = None
    return dataset, max_node_num

# import os.path as osp
# import re
# import shutil
# from torch_geometric.datasets import TUDataset
# from torch_geometric.utils import degree
# import torch
# import csv
# from collections import Counter

# from datasets.feature_expansion import FeatureExpander


# def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None):
#     if root is None or root == '':
#         path = osp.join(osp.expanduser('~'), 'pyG_data', name)
#     else:
#         path = osp.join(root)
#         if osp.exists(osp.join(path, name)) is False:
#             dataset = TUDataset(path, name)
#     degrees = feat_str.find("deg") >= 0
#     onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
#     onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None

#     dir_name = osp.join(path, name)
#     if osp.exists(dir_name) is True:
#         shutil.rmtree(osp.join(dir_name, 'processed'))
#     # Obtain max_node_num
#     with open(osp.join(path, name, "raw", name + "_graph_indicator.txt")) as f:
#         temp = list(csv.reader(f))
#     out = [int(i[0]) for i in temp]
#     max_node_num = Counter(out).most_common(1)[0][1]
#     # only maintain the 1000 nodes with the maximum degree
#     if max_node_num > 1000:
#         max_node_num = 1000
#     with open(osp.join(path, name, "raw", name + "_A.txt")) as f:
#         temp = list(csv.reader(f))
#     out = [(int(i[0]), int(i[1])) for i in temp]
#     out = torch.tensor(out).transpose(0, 1)

#     row, col = out
#     num_nodes = row.max().item()
#     deg = degree(row - 1, num_nodes)
#     max_deg = deg.max()
#     if max_deg < 10:
#         onehot_maxdeg = 10
#     elif max_deg < 50:
#         onehot_maxdeg = 50
#     pre_transform = FeatureExpander(max_node_num=max_node_num,
#                                     degree=degrees, onehot_maxdeg=onehot_maxdeg).transform

#     dataset = TUDataset(
#         path, name, pre_transform=pre_transform,
#         use_node_attr=True)

#     dataset.data.edge_attr = None
#     return dataset, max_node_num
