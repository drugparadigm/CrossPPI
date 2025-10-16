import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Protein_dataset(InMemoryDataset):
    def __init__(self,
                 protein_type,
                 root="dataset/protein",
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        
        root = os.path.join("dataset/protein", protein_type)
            
        # Protein dataset
        excel_file_path = "data/PPB-Affinity-Modified(pkd).xlsx"
        try:
            self.df = pd.read_excel(excel_file_path)
        except FileNotFoundError:
            logger.error(f"Excel file not found: {excel_file_path}")
            raise
        
        # Single contact map folder
        self.concat_folder_path = 'data/contact_maps_receptor'

        # Language model embedding folder
        self.emb_folder_path = 'data/embeddings_receptor'
        
        super().__init__(root, transform, pre_transform, pre_filter)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except FileNotFoundError:
            logger.warning(f"Processed data not found at {self.processed_paths[0]}. Will process data.")
            self.process()

    @property
    def processed_file_names(self):
        return "data_protein_receptor.pt"

    def process(self):
        data_list = []
        skipped_entries = []

        for index, row in self.df.iterrows():
            id_value = row['Complex ID']
            sequence = row['Receptor Chains'][:999] if len(row['Receptor Chains']) > 1000 else row['Receptor Chains']
            target_id = row["PDB"]

            # Check for contact map in single folder
            file_path = os.path.join(self.concat_folder_path, f"{id_value}.txt")
            if not os.path.exists(file_path):
                logger.warning(f"Contact map not found for Entry_ID: {id_value}")
                skipped_entries.append(id_value)
                continue

            try:
                # Load contact map (already binarized)
                matrix = np.loadtxt(file_path)
                edges = np.argwhere(matrix == 1)

                # Convert sequence to one-hot encoding
                try:
                    one_hot_sequence = [char_to_one_hot(char) for char in sequence]
                except KeyError as e:
                    logger.warning(f"Invalid character in sequence for Entry_ID: {id_value}, {e}")
                    skipped_entries.append(id_value)
                    continue

                # Load language model embedding
                emb_file_path = os.path.join(self.emb_folder_path, f"{target_id}.npy")
                if not os.path.exists(emb_file_path):
                    logger.warning(f"Embedding not found for Ligand_Protein_ID: {target_id}")
                    skipped_entries.append(id_value)
                    continue

                protein_emb = torch.tensor(np.load(emb_file_path), dtype=torch.float32)

                # Create data object
                x = torch.tensor(one_hot_sequence, dtype=torch.float32)
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                y = row['KD(M)']
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    t_id=row['Target_Protein_ID'],
                    e_id=row['Entry_ID'],
                    emb=protein_emb,
                    protein_len=x.size()[0]
                )
                data_list.append(data)

            except Exception as e:
                logger.error(f"Error processing Entry_ID: {id_value}, Error: {str(e)}")
                skipped_entries.append(id_value)
                continue

        if skipped_entries:
            logger.info(f"Skipped {len(skipped_entries)} entries due to missing data or errors: {skipped_entries}")

        if not data_list:
            logger.error("No valid data entries processed. Cannot save empty dataset.")
            raise ValueError("No valid data entries processed.")

        data, slices = self.collate(data_list)
        logger.info(f"Processed {len(data_list)} valid protein entries. Saving to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])

# Amino acid to one-hot
def char_to_one_hot(char):
    mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20  # Unknown or non-standard amino acid
    }
    return [mapping.get(char, 20)]  # Default to 'X' for unrecognized characters