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
import pandas as pd
from data.protein_processor import ProteinInference

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128

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
def process_protein_pairs(input_csv, output_csv, model_path="save/weights/model_cv_(updated)2_2_1.pth"):
    # Load the model
    model = PPI().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    required_columns = ['PDB', 'Ligand Chains', 'Receptor Chains']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")
    
    # Lists to store results
    results = []
    
    # Process each row
    for index, row in df.iterrows():
        pdb_id = row['PDB']
        ligand = row['Ligand Chains']
        receptor = row['Receptor Chains']
        print(pdb_id)
        
        # Process ligand and receptor sequences
        process_ligand = ProteinInference(sequence=ligand)
        processed_ligand = process_ligand.process()
        process_receptor = ProteinInference(sequence=receptor)
        processed_receptor = process_receptor.process()
        
        # Get prediction
        with torch.no_grad():
            prediction = model(processed_ligand.to(device), processed_receptor.to(device)).item()
        
        # Store results
        results.append({
            'PDB': pdb_id,
            'Ligand Chains': ligand,
            'Receptor Chains': receptor,
            'pKD': prediction
        })
    
    # Create output DataFrame and save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    # Example usage
    input_csv = "PPI(test).csv"
    output_csv = "PPI_out.csv"
    process_protein_pairs(input_csv, output_csv)
