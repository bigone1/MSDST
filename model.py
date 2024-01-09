import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import torch

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        # output = torch.sparse.mm(adj, support)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.hgcn_dim=hgcn_dim
        self.en1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.en2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.en3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.de3 = GraphConvolution(hgcn_dim[0],in_dim)
        self.de2 = GraphConvolution(hgcn_dim[1], hgcn_dim[0])
        self.de1 = GraphConvolution(hgcn_dim[2], hgcn_dim[1])
        self.dropout = dropout
    def forward(self, x, adj):
        x = self.en1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.en2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.en3(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.de1(x, adj)
        x1 = F.leaky_relu(x1, 0.25)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.de2(x1, adj)
        x1 = F.leaky_relu(x1, 0.25)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.de3(x1, adj)
        x1 = F.leaky_relu(x1, 0.25)
        return x,x1
class VCDN(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_cls, num_cls//4),
            nn.LeakyReLU(0.25),
            nn.Linear(num_cls//4, num_cls//16),
            nn.LeakyReLU(0.25),
            nn.Linear(num_cls//16, 1),
            nn.Sigmoid()
        )
        self.model.apply(xavier_init)
    def forward(self, in_list):
        num_view = len(in_list)
        he=torch.tensor([]).cuda()
        for i in range(num_view):
            he=torch.cat((he,in_list[i]),dim=1)
        output = self.model(he)
        return output
class GTN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm,gcn_drop,sample_weight):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.gcn_drop=gcn_drop
        self.sample_weight=sample_weight
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        # self.linear1 = nn.Linear(self.w_out[-1]*self.num_channels, self.w_out[-1])
        # self.linear2 = nn.Linear(self.w_out[-1], self.num_class)
        self.loss=nn.MSELoss()
        self.GCN=GCN_E(w_in,w_out,dropout=self.gcn_drop)
    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_
    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).cuda())
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).cuda()) + torch.eye(H.shape[0]).type(torch.FloatTensor).cuda()
        deg = torch.sum(H, dim=1)+1e-6
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor).cuda()
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H
    def forward(self, A, X):
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        H[0] = self.norm(H[0], add=True)
        X_,yuan=self.GCN(X,H[0].t())
        # X0 = self.linear1(X_)
        # X0 = F.relu(X0)
        # y = self.linear2(X0[target_x])
        loss=F.mse_loss(yuan,X,reduction='mean')
        return loss.cuda(),yuan.cuda(),X_.cuda()
class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W
class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
