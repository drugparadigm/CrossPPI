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
from data.protein_processor import ProteinInference
import random

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128

# PPI model definition
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
# Initialize model
model1 = PPI().to(device)
model2 = PPI().to(device)
model3 = PPI().to(device)
model4 = PPI().to(device)
model5 = PPI().to(device)
model1.load_state_dict(torch.load("save/model_cv_(t300(5_fold))2_1_1.pth", map_location=torch.device(device)))
model2.load_state_dict(torch.load("save/model_cv_(t300(5_fold))2_2_1.pth", map_location=torch.device(device)))
model3.load_state_dict(torch.load("save/model_cv_(t300(5_fold))2_3_1.pth", map_location=torch.device(device)))
model4.load_state_dict(torch.load("save/model_cv_(t300(5_fold))2_4_1.pth", map_location=torch.device(device)))
model5.load_state_dict(torch.load("save/model_cv_(t300(5_fold))2_5_1.pth", map_location=torch.device(device)))
model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

# Input sequences
ligand = "MGDKPIWEQIGSSFIQHYYQLFDNDRTQLGAIYIDASCLTWEGQQFQGKAAIVEKLSSLPFQKIQHSITAQDHQPTPDSCIISMVVGQLKADEDPIMGFHQMFLLKNINDAWVCTNDMFRLALHNFG"
receptor = "MAAQGEPQVQFKLVLVGDGGTGKTTFVKRHLTGEFEKKYVATLGVEVHPLVFHTNRGPIKFNVWDTAGQEKFGGLRDGYYIQAQCAIIMFDVTSRVTYKNVPNWHRDLVRVCENIPIVLCGNKVDIKDRKVKAKSIVFHRKKNLQYYDISAKSNYNFEKPFLWLARKLIGDPNLEFVAMPALAPPEVVMDPALAAQYEHDLEVAQTTALPDEDDDL"

# Process sequences
process_ligand = ProteinInference(sequence=ligand)
processed_ligand = process_ligand.process()
process_receptor = ProteinInference(sequence=receptor)
processed_receptor = process_receptor.process()

# Print processed data (for debugging)
print(processed_ligand)
print(processed_receptor)

# Run inference
o1 = model1(processed_ligand.to(device), processed_receptor.to(device)).item()
o2 = model2(processed_ligand.to(device), processed_receptor.to(device)).item()
o3 = model3(processed_ligand.to(device), processed_receptor.to(device)).item()
o4 = model4(processed_ligand.to(device), processed_receptor.to(device)).item()
o5 = model5(processed_ligand.to(device), processed_receptor.to(device)).item()

print((o1+o2+o3+o4+o5)/5)
