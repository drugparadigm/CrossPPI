import torch
from torch.utils.data import Dataset, DataLoader
from data import Protein_dataset
from model import Protein_feature_extraction,cross_attention
from torch_geometric.loader import DataLoader
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable
import numpy as np
import os
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# hyperparameter
BATCH_SIZE = 32
EPOCH = 400
hidden_dim = 128
seed_dataset = 2
seed = 1
LR = 5e-4

# ligand and receptor dataset
ligand_dataset = Protein_dataset('ligands')
receptor_dataset = Protein_dataset('receptor')

# set random seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=20)
set_seed(seed)

# combine two pyg dataset
class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)  

# stratified CV
class regressor_stratified_cv:
    def __init__(self, n_splits=10, n_repeats=2, group_count=10, random_state=0, strategy='quantile'):
        self.group_count = group_count
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer = KBinsDiscretizer(n_bins=self.group_count, encode='ordinal', strategy=self.strategy)  
            
    def split(self, X, y, groups=None):
        kgroups = self.discretizer.fit_transform(y[:, None])[:, 0]
        return self.cv.split(X, kgroups, groups)
    
    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


class PPI(nn.Module):
    def __init__(self):
        super(PPI, self).__init__()
        # Protein graph + seq
        self.ligand_graph_model = Protein_feature_extraction(hidden_dim)
        self.receptor_graph_model = Protein_feature_extraction(hidden_dim)
        # Cross fusion module
        self.cross_attention = cross_attention(hidden_dim)
        
        self.line1 = nn.Linear(hidden_dim * 2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.ligand1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.receptor1 = nn.Linear(hidden_dim, hidden_dim * 4)
        
        self.ligand2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.receptor2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, ligand_batch, receptor_batch):
        ligand_out_seq, ligand_out_graph, ligand_mask_seq, ligand_mask_graph, ligand_seq_final, ligand_graph_final = self.ligand_graph_model(ligand_batch, device)
        receptor_out_seq, receptor_out_graph, receptor_mask_seq, receptor_mask_graph, receptor_seq_final, receptor_graph_final = self.receptor_graph_model(receptor_batch, device)
        
        context_layer, attention_score = self.cross_attention(
            [ligand_out_seq, ligand_out_graph, receptor_out_seq, receptor_out_graph],
            [ligand_mask_seq, ligand_mask_graph, receptor_mask_seq, receptor_mask_graph],
            device
        )

        out_ligand = context_layer[-1][0]  # Shape: (batch_size, 2 * max_nodes, 128)
        out_receptor = context_layer[-1][1]  # Shape: (batch_size, 2 * max_nodes, 128)
        
        # Concatenate masks to match out_ligand's node dimension
        ligand_mask_combined = torch.cat((ligand_mask_seq, ligand_mask_graph), dim=1)  # Shape: (batch_size, 2 * max_nodes)
        receptor_mask_combined = torch.cat((receptor_mask_seq, receptor_mask_graph), dim=1)  # Shape: (batch_size, 2 * max_nodes)
        
        # Affinity Prediction Module
        ligand_cross_seq = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_seq_final) / 2
        ligand_cross_stru = ((out_ligand * ligand_mask_combined.unsqueeze(dim=2)).mean(dim=1) + ligand_graph_final) / 2        

        ligand_cross = (ligand_cross_seq + ligand_cross_stru) / 2
        ligand_cross = self.ligand2(self.dropout(self.relu(self.ligand1(ligand_cross))))

        receptor_cross_seq = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_seq_final) / 2
        receptor_cross_stru = ((out_receptor * receptor_mask_combined.unsqueeze(dim=2)).mean(dim=1) + receptor_graph_final) / 2
        
        receptor_cross = (receptor_cross_seq + receptor_cross_stru) / 2
        receptor_cross = self.receptor2(self.dropout(self.relu(self.receptor1(receptor_cross))))   
        
        out = torch.cat((ligand_cross, receptor_cross), 1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        
        return out

# 10 fold
kf = regressor_stratified_cv(n_splits=2, n_repeats=1, random_state=seed_dataset, group_count=5, strategy='uniform')

fold = 0
p_list = []
m_list = []
s_list = []
r_list = []
mae_list = []  # List to store MAE for each fold

for train_id, test_id in kf.split(ligand_dataset, ligand_dataset.y):
    max_p = -1
    max_s = -1
    max_rmse = 0
    max_mae = float('inf')  # Initialize max_mae to infinity (since lower MAE is better)
    fold = fold + 1

    print("Fold", fold)
    
    # Combine ligand Dataset and receptor Dataset
    train_dataset = CustomDualDataset(ligand_dataset[train_id], receptor_dataset[train_id])
    test_dataset = CustomDualDataset(ligand_dataset[test_id], receptor_dataset[test_id])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, shuffle=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=1, drop_last=False, shuffle=False
    )
    
    model = PPI().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
    optimal_loss = 1e10
    loss_fct = torch.nn.MSELoss()

    for epo in range(EPOCH):
        # train
        train_loss = 0
        for step, (batch_ligand, batch_receptor) in enumerate(train_loader):
            optimizer.zero_grad()
            pre = model(batch_ligand.to(device), batch_receptor.to(device))
            loss = loss_fct(pre.squeeze(dim=1), batch_ligand.y)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss
        # test
        with torch.set_grad_enabled(False):
            test_loss = 0
            model.eval()
            y_label = []
            y_pred = []
            for step, (batch_ligand_test, batch_receptor_test) in enumerate(test_loader):
                label = Variable(torch.from_numpy(np.array(batch_ligand_test.y))).float()
                score = model(batch_ligand_test.to(device), batch_receptor_test.to(device))
                n = torch.squeeze(score, 1)
                logits = torch.squeeze(score).detach().cpu().numpy()
                label_ids = label.to('cpu').numpy()
                loss_t = loss_fct(n.cpu(), label)
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()
                test_loss = test_loss + loss_t
            
        model.train()
        p = pearsonr(y_label, y_pred)
        s = spearmanr(y_label, y_pred)
        rmse = np.sqrt(mean_squared_error(y_label, y_pred))
        mae = mean_absolute_error(y_label, y_pred)  # Calculate MAE
        if max_p < p[0]:
            print('epo:', epo, 'pcc:', p[0], 'scc:', s[0], 'rmse:', rmse, 'mae:', mae)
            max_p = p[0]
            max_s = s[0]
            max_rmse = rmse
            max_mae = mae  # Update max_mae when PCC improves
            torch.save(model.state_dict(), 'save/' + 'model_cv_(updated)' + str(seed_dataset) + '_' + str(fold) + '_' + str(seed) + '.pth')
            with open("metrics(updated).txt",'a') as f:
                f.write(str(epo)+" "+str(p[0])+" "+str(s[0])+" "+str(rmse)+" "+str(mae)+" "+str(fold)+'\n')
        else:
            print("Not Saved",'epo:',epo, 'pcc:',p[0],'scc: ',s[0], 'rmse:',rmse, 'mae:',mae)
        p_list.append(max_p)
        r_list.append(max_rmse)
        s_list.append(max_s)
        mae_list.append(max_mae)  # Append the best MAE for this fold

    print('p:', np.mean(p_list))
    print('s:', np.mean(s_list))
    print('rmse:', np.mean(r_list))
    print('mae:', np.mean(mae_list))  # Print mean MAE across folds
