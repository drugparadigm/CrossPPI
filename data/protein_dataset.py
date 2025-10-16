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
        
        if protein_type not in ['ligands', 'receptor']:
            logger.error(f"Invalid protein_type: {protein_type}. Must be 'ligand' or 'receptor'.")
            raise ValueError("protein_type must be 'ligand' or 'receptor'.")
        
        self.protein_type = protein_type
        root = os.path.join("dataset/protein", protein_type)
            
        # Protein dataset
        excel_file_path = "data/PPB-Affinity-Modified(pkd).xlsx"
        try:
            self.df = pd.read_excel(excel_file_path)
        except FileNotFoundError:
            logger.error(f"Excel file not found: {excel_file_path}")
            raise
        
        # Contact map and embedding folders based on protein_type
        self.concat_folder_path = f'data/contact_maps_{protein_type}'
        self.emb_folder_path = f'data/embeddings_{protein_type}'
        
        super().__init__(root, transform, pre_transform, pre_filter)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except FileNotFoundError:
            logger.warning(f"Processed data not found at {self.processed_paths[0]}. Will process data.")
            self.process()

    @property
    def processed_file_names(self):
        return f"data_protein_{self.protein_type}.pt"

    def process(self):
        data_list = []
        skipped_entries = []

        for index, row in self.df.iterrows():
            id_value = row['Complex ID']
            sequence_key = 'Ligand Chains' if self.protein_type == 'ligands' else 'Receptor Chains'
            sequence = row[sequence_key][:998] if len(row[sequence_key]) > 999 else row[sequence_key]
            target_id = row["Complex ID"]

            # Check for contact map in single folder
            file_path = os.path.join(self.concat_folder_path, f"{id_value}.txt")
            if not os.path.exists(file_path):
                logger.warning(f"Contact map not found for Entry_ID: {id_value}")
                skipped_entries.append(id_value)
                continue

            try:
                # Load contact map (already binarized)
                matrix = np.loadtxt(file_path)
                #edges = np.argwhere(matrix == 1)
                sequence_length=len(sequence)
                if matrix.shape != (sequence_length, sequence_length):
                    logger.error(f"Contact map shape {matrix.shape} does not match sequence length {sequence_length} for Entry_ID: {id_value}")
                    skipped_entries.append(id_value)
                    continue
                edges = np.argwhere(matrix == 1)
                if edges.size > 0 and (edges.max() >= sequence_length):
                    logger.error(f"Contact map contains invalid indices (max {edges.max()}) for sequence length {sequence_length}, Entry_ID: {id_value}")
                    skipped_entries.append(id_value)
                    continue
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
                    t_id=target_id,
                    e_id=id_value,
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
        logger.info(f"Processed {len(data_list)} valid {self.protein_type} entries. Saving to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])

# Amino acid to one-hot
def char_to_one_hot(char):
    mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20  # Unknown or non-standard amino acid
    }
    return [mapping.get(char, 20)]  # Default to 'X' for unrecognized characters