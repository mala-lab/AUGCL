import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

def getclusters(x, x_aug):
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = sim_matrix.detach().cpu().numpy()
    label_matrix = np.zeros([batch_size, batch_size])
    for i in range(batch_size):
        clf = KMeans(n_clusters=2)
        kdata = sim_matrix[i].reshape(-1,1)
        clf.fit(kdata)
        labels = clf.labels_
        if labels[i] != 1:
            labels = np.abs(labels-1)
        label_matrix[i] = labels
    return label_matrix

class uncermodel(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(uncermodel, self).__init__()
        self.layer1 = torch.nn.Linear(n_feature, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x