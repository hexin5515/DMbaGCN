import torch
import math
import numpy as np
import torch
import torch.nn.functional as F


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    
    return data

def rand_train_test_idx(data, train_prop=.6, valid_prop=.2, seed=12134):
    """ randomly splits label into train/valid/test splits """
    index=[i for i in range(0, data.y.shape[0])]
    n = data.y.shape[0]
    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(index, int(n * train_prop), replace=False)
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,int(n * valid_prop),replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)

    return data

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

def info_nce_loss(z_deep, z_shallow, tau=0.2):
    """
    Mutual Information (InfoNCE) Loss
    ---------------------------------
    对齐相同节点在浅层和深层的表示(正样本)，其他节点为负样本。
    z_deep:   Tensor [N, d] 深层节点表示
    z_shallow:Tensor [N, d] 浅层节点表示
    tau: 温度参数
    """
    z_d = F.normalize(z_deep, dim=1)
    z_s = F.normalize(z_shallow, dim=1)

    sim = torch.mm(z_d, z_s.t()) / tau

    target = torch.arange(sim.size(0), device=z_d.device)

    return F.cross_entropy(sim, target)

def random_sample_edges(adj, n, exclude):
        itr = sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

def sample_forever(adj, exclude):
    """Randomly random sample edges from adjacency matrix, `exclude` is a set
    which contains the edges we do not want to sample and the ones already sampled
    """
    while True:
        t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
        if t not in exclude:
            yield t
            exclude.add(t)
            exclude.add((t[1], t[0]))

