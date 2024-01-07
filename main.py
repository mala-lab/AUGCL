import torch
import ipdb
import random
import json
import numpy as np
import os.path as osp
import torch.nn as nn
from gin import Encoder
from torch._C import device
import torch.nn.functional as F
from aug import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader
from evaluate_embedding import evaluate_embedding
from arguments import arg_parse
from uncertaintymodel import uncermodel, getclusters

import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class simclr(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers):
        super(simclr, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self,  data, x, edge_index, batch, num_graphs):
        self.data = data
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss

    def weightedloss(self, x, x_aug, uncertainty):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        reweight = uncertainty.detach()
        reweight = reweight / reweight.mean(dim=1, keepdim=True)
        reweight.fill_diagonal_(1)
        sim_matrix = reweight * sim_matrix
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def CalculateUncertainty(model, optimizer_un, x, x_aug, batchsize, reward, train=True, device='cpu'):
    trainlabels = getclusters(x, x_aug)
    trainlabels = torch.tensor(trainlabels).type(torch.LongTensor).to(device)
    num_sample, dims = x.shape
    x, x_aug = x.detach(), x_aug.detach()
    x = x.unsqueeze(1).repeat(1,num_sample,1)
    x_aug = x_aug.unsqueeze(0).repeat(num_sample,1,1)
    traindata = torch.cat([x, x_aug], dim=2)
    traindata = traindata.reshape(-1, dims)
    if train:
        trainlabels = trainlabels.reshape(-1,1)
        model.train()
        torch_dataset = Data.TensorDataset(traindata.cpu(), trainlabels.cpu())
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batchsize, shuffle=True, num_workers=2,)
        lossall = 0  
        for step, (batch_x, batch_y) in enumerate(loader): 
            optimizer_un.zero_grad()
            outputs = model(batch_x.to(device))
            outputs = F.softmax(outputs, dim=1)
            outputs, reservation = outputs[:, :-1], outputs[:, -1]
            gain = torch.gather(outputs, dim=1, index=batch_y.to(device)).squeeze()
            doubling_rate = (gain.add(reservation.div(reward))).log()
            uncer_loss = -doubling_rate.mean()
            uncer_loss.backward()
            optimizer_un.step()
            lossall +=uncer_loss.item() * batch_x.shape[0]
        print("Uncertainty Estimation Loss: %.4f" % (lossall/traindata.shape[0]))
    else:
        model.eval()
        outputs = model(traindata)
        outputs = F.softmax(outputs, dim=1)
        outputs, uncertainty = outputs[:, :-1], outputs[:, -1]
        return uncertainty, trainlabels


def run(args,seed,epochs, log_interval):
    args = arg_parse()
    setup_seed(seed)
    accuracies = {'val': [], 'test': []}
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    model = simclr(dataset_num_features, args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    uncermodel = uncermodel(args.hidden_dim*args.num_gc_layers, 128, 3).to(device)
    optimizer_un = torch.optim.Adam(uncermodel.parameters(), lr=0.01)
    
    print('================')
    print('lr: {}'.format(lr))
    print('Dataset Size: {}'.format(len(dataset)))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')
    #pretrain a network to obtain embeddings of input graphs
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()
            data = data.to(device)
            data_aug = data_aug.to(device)
            x = model(data, data.x, data.edge_index, data.batch, data.num_graphs)
            x_aug = model(data_aug, data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            loss = model.loss(x, x_aug)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()


    # train an uncertainty estimation model
    model.eval()
    for i in range(10):
        for data in dataloader:
            data, data_aug = data
            data = data.to(device)
            data_aug = data_aug.to(device)
            x = model(data, data.x, data.edge_index, data.batch, data.num_graphs)
            x_aug = model(data_aug, data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            CalculateUncertainty(uncermodel, optimizer_un, x, x_aug, batchsize=2048, reward=args.reward, train=True, device=device)

    # further train the network with uncertainty
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()
            data = data.to(device)
            data_aug = data_aug.to(device)
            x = model(data, data.x, data.edge_index, data.batch, data.num_graphs)
            x_aug = model(data_aug, data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            unceroutputs, _ = CalculateUncertainty(uncermodel, optimizer_un, x, x_aug, batchsize=2048, reward=args.reward, train=False, device=device)
            unceroutputs = unceroutputs.reshape(x.shape[0],x.shape[0])
            loss = model.weightedloss(x, x_aug, unceroutputs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataset)))
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval, device)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            print(accuracies['val'][-1], accuracies['test'][-1])

    return accuracies['val'][-1], accuracies['test'][-1]    



if __name__ == '__main__':
    args = arg_parse()
    seednum = 5
    acc = {'val': [], 'test': [], 'valmeanstd':[], 'testmeanstd':[]}
    epochs = 20
    log_interval = 2
    for seed in range(seednum):
        acc_val, acc_test = run(args, seed, epochs, log_interval)
        print('seed:%d val:%.4f test:%.4f' % (seed, acc_val, acc_test))
        acc['val'].append(acc_val)
        acc['test'].append(acc_test)

    valmean, valstd = np.mean(acc['val'])*100, np.std(acc['val'])*100
    testmean, teststd = np.mean(acc['test'])*100, np.std(acc['test'])*100
    acc['valmeanstd'].append(valmean)
    acc['valmeanstd'].append(valstd)
    acc['testmeanstd'].append(testmean)
    acc['testmeanstd'].append(teststd)
    print('--------finish--------')
    print('val mean:%.4f val std:%.4f' %(valmean, valstd) )
    print('test mean:%.4f test std:%.4f' %(testmean, teststd))

    with open('logs/log_out_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(acc)
        f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, args.lr, s))
        f.write('\n')
