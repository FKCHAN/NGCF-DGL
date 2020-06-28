import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from sklearn import manifold
from torch import autograd

from NGCF.NGCF_GCN import NGCF_GCN
from NGCF.Dataset import NaiveGraphDataset
from NGCF.Config import config
from dgl import function as fn
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate(model, features, labels, mask):  # TODO：修改评估方法
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def graph_loss(model, graph, lam):
    def message_func(edges):
        dst_data = edges.dst['h']
        src_data = edges.src['h']
        return_data = torch.neg(torch.log(torch.sigmoid(dst_data - src_data)))
        return {'loss': return_data}

    feat = graph.ndata['feat']
    graph.srcdata['h'] = feat
    graph.update_all(message_func,
                     fn.sum(msg='loss', out='total_loss'))
    loss1 = sum(sum(graph.dstdata['total_loss'])) / graph.number_of_edges()

    loss2 = 0
    for layer in model.layers:
        w1 = layer.weight1
        w2 = layer.weight2

        def norm2(lis):
            losss = 0
            for digit in lis:
                losss += digit ** 2
            return losss

        loss2 = sum(loss2 + norm2(w1) + norm2(w2)) / len(model.layers)
    loss2 = lam * loss2

    loss = loss1 + loss2

    return loss


def draw(logit_records, data):
    for logit in logit_records:
        '''t-SNE'''
        X = []
        y = []
        for vector in logit:
            _min, _max = float(min(vector)), float(max(vector))
            vec = []
            for digit in vector:
                vec.append((float(digit) - _min) / (_max - _min))
            X.append(vec)
        for i in range(data.n_users):
            y.append(1)
        for i in range(data.n_items):
            y.append(2)
        X = np.array(X)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(X)

        print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

        '''嵌入空间可视化'''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, marker='o', s=60,
                    cmap=plt.cm.Spectral)  # c--color，s--size,marker点的形状

        edges = data.graph.all_edges()
        for i in range(len(edges[0])):
            x1 = X_norm[edges[0][i], 0]
            y1 = X_norm[edges[0][i], 1]
            x2 = X_norm[edges[1][i], 0]
            y2 = X_norm[edges[1][i], 1]
            plt.plot([x1, x2], [y1, y2], color='skyblue', label='fta')

        plt.show()


def main(args):
    # load and preprocess dataset
    data = NaiveGraphDataset('gowalla')  # TODO 添加至配置
    features = torch.FloatTensor(data.features)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_edges = data.g.number_of_edges()
    print("""----Data statistics----'
      #Edges %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    g = data.g
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = NGCF_GCN(g,
                     config['dim_emb'],
                     args.n_layers,
                     F.leaky_relu,
                     args.dropout)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    print('----start training----')
    dur = []
    logits_record = []
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()

        # forward
        logits = model(features)
        if epoch % 20 == 0:
            logits_record.append(logits)
        loss = graph_loss(model, g, lam=0.001)  # TODO: 参数化

        optimizer.zero_grad()
        # autograd.grad(outputs=loss, inputs=features, allow_unused=True)
        loss.backward()
        optimizer.step()

        t1 = time.time() - t0
        dur.append(t1)

        # acc = evaluate(model, features, labels, val_mask)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                     acc, n_edges / np.mean(dur) / 1000))

        print('Epoch {}:\n'
              '\ttrain loss: {}\n'
              '\ttime spent: {}'.format(epoch, loss, t1))

    draw(logits_record, data)
    # print()
    # acc = evaluate(model, features, labels, test_mask)
    # print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NGCF_GCN')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=5e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=40,
                        help="number of training epochs")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
