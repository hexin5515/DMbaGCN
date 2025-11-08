import argparse

from ogb.nodeproppred import Evaluator
from dataset_loader import DataLoader
from utils import random_planetoid_splits, rand_train_test_idx, index_to_mask, info_nce_loss, random_sample_edges
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
from torch_geometric.datasets import Planetoid
import time
from torch_geometric.utils import to_dense_adj, add_self_loops, remove_self_loops
import os 

os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

def RunExp(args, dataset, data, Net, rb):

    def train(args, model, optimizer, data, dprate, rb):
        model.train()
        optimizer.zero_grad()
        if args.net not in ['GCN_mamba_Net_New']:
            out_main = model(data)[[data.train_mask]]
            data.y = torch.squeeze(data.y)
            nll_main = F.nll_loss(out_main, data.y[data.train_mask])
        else:
            all_layers_out_main, out_main = model(data.x, data.adj_t)
            data.y = torch.squeeze(data.y)
            nll_main = F.nll_loss(out_main[[data.train_mask]], data.y[data.train_mask])

        loss = nll_main
        reg_loss=None
        loss.backward()
        optimizer.step()
        del out_main

    def test(args, model, data, rb):
        model.eval()
        if args.net not in ['GCN_mamba_Net_New']:
            logits_main = model(data.x, data.adj_t)
        else:
            _, logits_main = model(data.x, data.adj_t)
        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            if args.dataset in ['Chameleon', 'Squirrel', 'Actor', 'Wisconsin']:
                pred = logits_main[mask[:,rb]].max(-1)[1]
                acc = pred.eq(data.y[mask[:,rb]]).sum().item() / mask[:,rb].sum().item()
                loss = F.nll_loss(logits_main[mask[:,rb]], data.y[mask[:,rb]])
            else:
                pred = logits_main[mask].max(-1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(logits_main[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses
    
    def normalize_adj_tensor(adj):
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')

    if args.dataset not in ['Chameleon', 'Squirrel', 'Actor', 'Wisconsin']:
        permute_masks = rand_train_test_idx
        data = permute_masks(data, seed=15)
    data = data.to(device)
    if args.net == 'Mamba_Net':
        tmp_net = Net(data, dataset, args)
    else:
        tmp_net = Net(dataset, args)

    model = tmp_net.to(device)


    if args.net == 'GCN_mamba_Net_New':
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.net == 'GCNII':
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=args.weight_decay1),
            dict(params=model.non_reg_params, weight_decay=args.weight_decay2)
        ], lr=args.lr)
    elif args.net=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(args, model, optimizer, data, args.dprate, rb)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(args, model, data, rb)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_acc_history[-(args.early_stopping + 1):-1])
                if val_acc < tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break
        print('train_acc:{},val_acc:{},temp_test_acc:{}'.format(train_acc, val_acc, tmp_test_acc))
    return test_acc, best_val_acc, theta, time_run, model

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=5000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')     
    parser.add_argument('--Stat_lr', type=float, default=0.05, help='State learning rate.')   
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay.') 
    parser.add_argument('--mamba_weight_decay', type=float, default=1e-4, help='weight decay.') 
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout for neural networks.')

    parser.add_argument('--lamda', type=float, default=0.5, help='propagation steps.')
    parser.add_argument('--weight_decay1', type=float, default=5e-4, help='weight decay.')
    parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay.')

    parser.add_argument('--dprate', type=float, default=0.2, help='dropout for propagation layer.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')

    parser.add_argument('--dataset', type=str, choices=['ogbn-proteins','ogbn-arxiv','AMiner', 'Reddit', 'CS', 'Physics', 'Corafull','Wisconsin', 'Cora','Citeseer','Pubmed','Computers','Photo','Chameleon','Squirrel','Actor','Texas','Cornell',
                                                        'Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions'],
                        default='Cora')
    parser.add_argument('--device', type=int, default=1, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10**6, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GT_Net', 'SSGC_Net', 'SGC_Net', 'GCNII', 'GCN_mamba_Net_New', 'GCN_mamba_Net', 'GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN','BernNet','MLP', 'Mamba_Net'], default='BernNet')

    # parameters for KAN
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--graph_weight', type=float, default=0.9)

    # parameters for mamba
    parser.add_argument('--window_size', type=int, default=10, help='')
    parser.add_argument('--headdim', type=int, default=16, help='hidden dim.')
    parser.add_argument('--d_model', type=int, default=32, help='hidden units.')
    parser.add_argument('--d_inner', type=int, default=32, help='')
    parser.add_argument('--dt_rank', type=int, default=4, help='')
    parser.add_argument('--d_state', type=int, default=64, help='')
    parser.add_argument('--bias', action='store_true', help='Use bias if set')
    parser.add_argument('--mamba_dropout', type=float, default=0.8, help='')
    parser.add_argument('--layer_num', type=int, default=3, help='')
    parser.add_argument('--d_conv', type=int, default=4, help='')
    parser.add_argument('--expand', type=int, default=2, help='')

    args = parser.parse_args()
    print(args.bias)
    fix_seed(args.seed)
    SEEDS = random.sample(range(10**9, 4294967296), 10**6)


    print(args)
    print("---------------------------------------------")

    dataset = DataLoader(args.dataset)
    data = dataset[0]

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
    adj_t = adj_t + SparseTensor.eye(data.num_nodes).to(device)
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t
    data.adj_perturb = adj_t

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    if args.net == 'SGC_Net':
        x = data.x
        for i in range(32):
            x = torch.mm(adj_t.to_dense(), x)
        data.x = x
    elif args.net == 'SSGC_Net':
        alpha = 0.05
        x = data.x
        emb = alpha * x
        degree = 2
        for i in range(degree):
            x = torch.mm(adj_t.to_dense(), x)
            emb = emb + (1-alpha)*x/degree
        data.x = emb

    results = []
    thetas = []
    time_results=[]
    for RP in tqdm(range(args.runs)):
        if args.dataset not in ['ogbn-arxiv']:
            fix_seed(SEEDS[RP])
        args.seed = SEEDS[RP]
        gnn_name = args.net
        if gnn_name =='GCN_mamba_Net_New':
            Net = GCN_mamba_Net
        if gnn_name == 'GCNII':
            Net = GCNII
        if gnn_name == 'GPRGNN':
            Net = GPRGNN
        if gnn_name == 'APPNP':
            Net = APPNP_Net


        test_acc, best_val_acc, theta_0, time_run,tmp_net = RunExp(args, dataset, data, Net, RP)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        thetas.append(theta_0)
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')

    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    print("each run avg_time:",run_sum/(args.runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.2f}±{uncertainty*100:.2f}  \t val acc mean = {val_acc_mean:.2f}')