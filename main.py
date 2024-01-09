import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from adj import cal_adj_mat_parameter,gen_adj_mat_tensor,cal_sample_weight
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from model import VCDN,GTN
import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import	concordance_index
import optuna
import random,time,csv
import warnings
warnings.filterwarnings('ignore')
def seed_torch(seed=520):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
def silhouette_plot(codes):
    estimator = KMeans(n_clusters=4, random_state=555)  # construct estimator
    estimator.fit(codes)
    return silhouette_score(codes, estimator.labels_, metric='cosine')
def silhouette_plot0(codes, max_k=6):
    Scores = []  # 存silhouette scores
    for k in range(2, max_k):
        estimator = KMeans(n_clusters=k, random_state=555)  # construct estimator
        estimator.fit(codes)
        Scores.append(
            silhouette_score(codes, estimator.labels_, metric='cosine'))
    return Scores
def silhouette_plot1(codes, max_k=11):
    Scores = []  # 存silhouette scores
    for k in range(2, max_k):
        estimator = KMeans(n_clusters=k, random_state=555)  # construct estimator
        estimator.fit(codes)
        Scores.append(
            silhouette_score(codes, estimator.labels_, metric='cityblock'))
    estimator = KMeans(n_clusters=2, random_state=555)  # construct estimator
    estimator.fit(codes)
    # return silhouette_score(codes, estimator.labels_, metric='cosine')
    return Scores,estimator.labels_
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        model_dict[module].cuda()    
    return model_dict
def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
def	CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    hazards	= hazards.cpu().numpy().reshape(-1)
    return(concordance_index(survtime_all, -hazards, labels))
def gen_trte_adj_mat(data_trte_list, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_list = []
    for i in range(len(data_trte_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_trte_list[i], adj_metric)
        adj_list.append(gen_adj_mat_tensor(data_trte_list[i], adj_parameter_adaptive, adj_metric))
    return adj_list
def init_model_dict(num_view, num_class, dim_list, dim_he_list1, dim_he_list2, dim_he_list3, sample_weight,gcn_dopout,num_edge=4):
    model_dict = {}
    model_dict["E{:}".format(1)] = GTN(num_edge=num_edge,num_channels=1,w_in=dim_list[0],w_out=dim_he_list1,num_class=2,num_layers=1,norm=True,gcn_drop=gcn_dopout,sample_weight=sample_weight)
    model_dict["E{:}".format(2)] = GTN(num_edge=num_edge,num_channels=1,w_in=dim_list[1],w_out=dim_he_list2,num_class=2,num_layers=1,norm=True,gcn_drop=gcn_dopout,sample_weight=sample_weight)
    model_dict["E{:}".format(3)] = GTN(num_edge=num_edge,num_channels=1,w_in=dim_list[2],w_out=dim_he_list3,num_class=2,num_layers=1,norm=True,gcn_drop=gcn_dopout,sample_weight=sample_weight)
    if num_view >= 2:
        dim_hc=dim_he_list1[2]+dim_he_list2[2]+dim_he_list3[2]
        model_dict["C"] = VCDN(dim_hc)
    return model_dict
def init_optim(num_view, model_dict, optimizer_name,lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["E{:}".format(i+1)] = getattr(optim,optimizer_name)(model_dict["E{:}".format(i+1)].parameters(), lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = getattr(optim,optimizer_name)(model_dict["C"].parameters(), lr=lr_c,weight_decay=0)
    return optim_dict
def train_epoch(data_list, adj, label, time, model_dict, optim_dict):
    c_index=None
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["E{:}".format(i+1)].zero_grad()
        ci_loss = model_dict["E{:}".format(i+1)](adj,data_list[i])[0].cuda()
        ci_loss.backward()
        optim_dict["E{:}".format(i+1)].step()
    if num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["E{:}".format(i+1)](adj,data_list[i])[2])
        c = model_dict["C"](ci_list)
        R_matrix_train = np.zeros([len(label), len(label)], dtype=int)
        for	i in range(len(label)):
            for	j in range(len(label)):
                R_matrix_train[i,j]	= time[j] >= time[i]
        train_R	= torch.FloatTensor(R_matrix_train)
        train_R	= train_R.cuda()
        theta = c.reshape(-1)
        exp_theta =	torch.exp(theta)
        c_loss= -torch.mean( (theta-torch.log(torch.sum( exp_theta*train_R ,dim=1))) * label.float())
        c_loss.backward()
        optim_dict["C"].step()
        c_index	= CIndex_lifeline(c.data, label, time)
        he=torch.tensor([])
        for i in range(num_view):
            he=torch.cat((he,ci_list[i].cpu()),dim=1)
        he=he.detach().numpy()
        ss=silhouette_plot(he)
    return c_index,ci_list,c.data,ss
def train_epoch1(data_list, adj, label, time, model_dict):
    loss_dict = {}
    c_index=None
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    if num_view >= 2:
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["E{:}".format(i+1)](adj,data_list[i])[2])
        c = model_dict["C"](ci_list)
        c_index	= CIndex_lifeline(c.data, label, time)
        he=torch.tensor([])
        for i in range(num_view):
            he=torch.cat((he,ci_list[i].cpu()),dim=1)
        he=he.detach().numpy()
        ss,labels=silhouette_plot1(he)
    return loss_dict,c_index,ci_list,c.data,ss,labels
def objective(trial):
    seed_torch()
    
    params={
            'adj_parameter':trial.suggest_int('adj_parameter',2,10),
            'dim_he_list1':[trial.suggest_int('dim_he_list1_0',4000,6016,step=32),trial.suggest_int('dim_he_list1_1',2000,4016,step=32),trial.suggest_int('dim_he_list1_2',600,1992,step=32)],
            'dim_he_list2':[trial.suggest_int('dim_he_list2_0',180,252,step=4),trial.suggest_int('dim_he_list2_1',100,180,step=4),trial.suggest_int('dim_he_list2_2',20,100,step=4)],
            'dim_he_list3':[trial.suggest_int('dim_he_list3_0',3000,5016,step=32),trial.suggest_int('dim_he_list3_1',1000,3016,step=32),trial.suggest_int('dim_he_list3_2',400,1008,step=32)],
            'optimizer_name':trial.suggest_categorical('optimizer_name',["Adadelta","Adam","RMSprop","SGD"]),
            'lr_e':trial.suggest_float('lr_e',1e-4,1e-2,step=1e-4),
            'lr_c':trial.suggest_float('lr_c',1e-4,1e-2,step=1e-4),
            'gcn_drop':trial.suggest_float('gcn_drop',0.1,0.5,step=0.1)
        }
    adj_list = gen_trte_adj_mat(data, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    dim_list = [x.shape[1] for x in data]
    model_dict = init_model_dict(3, 2, dim_list, params['dim_he_list1'],params['dim_he_list2'],params['dim_he_list3'], sample_weight,params["gcn_drop"])
    optim_dict = init_optim(3, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
    for m in model_dict:
        model_dict[m].cuda()
    num_epoch=400
    for epoch in range(num_epoch):
        c_index,ci_list,c,ss=train_epoch(data, A, status_tensor, time,model_dict, optim_dict)
        # trial.report(c_index,epoch)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
    return c_index,ss
def bestparam():
    seed_torch()
    clinical=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/clinical.csv',index_col=0)
    tpm=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/tpm.csv',index_col=0)
    mirna=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/mirna.csv',index_col=0)
    methy=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/methy.csv',index_col=0)
    tpm=tpm.transpose()
    mirna=mirna.transpose()
    methy=methy.transpose()
    data=[]
    data.append(torch.FloatTensor(tpm.values).cuda())
    data.append(torch.FloatTensor(mirna.values).cuda())
    data.append(torch.FloatTensor(methy.values).cuda())
    time=clinical.iloc[:,1:3].values[:,0]/30.
    status=clinical.iloc[:,1:3].values[:,1]
    status_tensor=torch.LongTensor(status).cuda()
    sample_weight = torch.FloatTensor(cal_sample_weight(status, 2)).cuda()
    status_tensor = status_tensor.cuda()
    sample_weight = sample_weight.cuda()
    params={
            'adj_parameter':7,
            'dim_he_list1':[5056,2640,1144],
            'dim_he_list2':[224,148,52],
            'dim_he_list3':[3416,1512,464],
            'optimizer_name':"RMSprop",
            'lr_e':0.0036,
            'lr_c':0.0013,
            'gcn_drop':0.3
        }
    adj_list = gen_trte_adj_mat(data, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    dim_list = [x.shape[1] for x in data]
    model_dict = init_model_dict(3, 2, dim_list, params['dim_he_list1'],params['dim_he_list2'],params['dim_he_list3'], sample_weight,params["gcn_drop"])
    optim_dict = init_optim(3, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
    for m in model_dict:
        model_dict[m].cuda()
    num_epoch=400
    for epoch in range(num_epoch):
        loss_dict,c_index,ci_list,c,ss=train_epoch(data, A, status_tensor, time,model_dict, optim_dict)
    save_model_dict('/home/zhoulin/shiyan/dst_subtype/model',model_dict)
    return c_index,ss

def load():
    seed_torch()
    clinical=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/clinical.csv',index_col=0)
    tpm=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/tpm.csv',index_col=0)
    mirna=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/mirna.csv',index_col=0)
    methy=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/methy.csv',index_col=0)
    tpm=tpm.transpose()
    mirna=mirna.transpose()
    methy=methy.transpose()
    data=[]
    data.append(torch.FloatTensor(tpm.values).cuda())
    data.append(torch.FloatTensor(mirna.values).cuda())
    data.append(torch.FloatTensor(methy.values).cuda())
    time=clinical.iloc[:,1:3].values[:,0]/30.
    status=clinical.iloc[:,1:3].values[:,1]
    status_tensor=torch.LongTensor(status).cuda()
    sample_weight = torch.FloatTensor(cal_sample_weight(status, 2)).cuda()
    status_tensor = status_tensor.cuda()
    sample_weight = sample_weight.cuda()
    params={
            'adj_parameter':7,
            'dim_he_list1':[5056,2640,1144],
            'dim_he_list2':[224,148,52],
            'dim_he_list3':[3416,1512,464],
            'optimizer_name':"RMSprop",
            'lr_e':0.0036,
            'lr_c':0.0013,
            'gcn_drop':0.3
        }
    adj_list = gen_trte_adj_mat(data, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    dim_list = [x.shape[1] for x in data]
    model_dict = init_model_dict(3, 2, dim_list, params['dim_he_list1'],params['dim_he_list2'],params['dim_he_list3'], sample_weight,params["gcn_drop"])
    for m in model_dict:
        model_dict[m].cuda()
    # num_epoch=500
    # for epoch in range(num_epoch):
    #     loss_dict,ci_list,ss=train_epoch(data, A, status_tensor, time,model_dict, optim_dict)
    model_dict=load_model_dict('/home/zhoulin/shiyan/dst_subtype/model', model_dict)
    loss_dict,c_index,ci_list,cdata,ss,labels=train_epoch1(data, A, status_tensor, time,model_dict)
    clinical['class']=labels
    clinical.to_csv('cc.csv')
    return ss
if __name__=='__main__':
    torch.set_num_threads(1)
    clinical=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/clinical.csv',index_col=0)
    tpm=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/tpm.csv',index_col=0)
    mirna=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/mirna.csv',index_col=0)
    methy=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/methy.csv',index_col=0)
    tpm=tpm.transpose()
    mirna=mirna.transpose()
    methy=methy.transpose()
    data=[]
    data.append(torch.FloatTensor(tpm.values).cuda())
    data.append(torch.FloatTensor(mirna.values).cuda())
    data.append(torch.FloatTensor(methy.values).cuda())
    time=clinical.iloc[:,1:3].values[:,0]/30.
    status=clinical.iloc[:,1:3].values[:,1]
    status_tensor=torch.LongTensor(status).cuda()
    sample_weight = torch.FloatTensor(cal_sample_weight(status, 2)).cuda()
    status_tensor = status_tensor.cuda()
    sample_weight = sample_weight.cuda()
    optunatrain=1
    if optunatrain:
        seed_torch()
        study=optuna.create_study(directions=['maximize','maximize'])
        study.optimize(objective,n_trials=1500,gc_after_trial=True)
        res=[]
        for i in range(len(study.get_trials())):
            tmp=[]
            tmp.append(study.get_trials()[i].values[0])
            tmp.append(study.get_trials()[i].params)
            res.append(tmp)
        out=open('dst_subtype.csv','w',newline="")
        csvwriter=csv.writer(out,dialect='excel')
        csvwriter.writerow(['c_index','parameters'])
        csvwriter.writerows(res)
        out.close()
    else:
        seed_torch()
        # print(bestparam())
        print(load())