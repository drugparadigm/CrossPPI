import torch
from torch_geometric.data import Data
import numpy as np
import logging
import esm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinInference:
    def __init__(self, sequence):
        self.sequence = sequence
        
        # Load ESM model
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

    def generate_esm_contact_map(self, sequence):
        """Generate contact map using ESM model with 0.5 threshold."""
        try:
            # Prepare sequence for ESM
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
                contact_map = results["contacts"][0].cpu().numpy()
            
            # Binarize contact map with 0.5 threshold
            binarized_map = (contact_map > 0.5).astype(np.int32)
            return binarized_map
        except Exception as e:
            logger.error(f"Error generating ESM contact map: {str(e)}")
            raise

    def generate_esm_embedding(self, sequence):
        """Generate protein embedding using ESM model."""
        try:
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                embedding = results["representations"][33][0].cpu().numpy()
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating ESM embedding: {str(e)}")
            raise

    def process(self):
        """Process single sequence for inference."""
        try:
            sequence = self.sequence[:998] if len(self.sequence) > 999 else self.sequence
            
            # Generate contact map using ESM
            contact_map = self.generate_esm_contact_map(sequence)
            sequence_length = len(sequence)
            
            if contact_map.shape != (sequence_length, sequence_length):
                logger.error(f"Contact map shape {contact_map.shape} does not match sequence length {sequence_length}")
                raise ValueError("Contact map shape mismatch")
            
            edges = np.argwhere(contact_map == 1)
            if edges.size > 0 and (edges.max() >= sequence_length):
                logger.error(f"Contact map contains invalid indices (max {edges.max()}) for sequence length {sequence_length}")
                raise ValueError("Invalid contact map indices")
            
            # Convert sequence to one-hot encoding
            try:
                one_hot_sequence = [char_to_one_hot(char) for char in sequence]
            except KeyError as e:
                logger.error(f"Invalid character in sequence: {e}")
                raise
            
            # Generate ESM embedding
            protein_emb = torch.tensor(self.generate_esm_embedding(sequence), dtype=torch.float32)
            
            # Create data object
            x = torch.tensor(one_hot_sequence, dtype=torch.float32)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            data = Data(
                x=x,
                edge_index=edge_index,
                emb=protein_emb,
                protein_len=x.size()[0]
            )
            
            logger.info(f"Successfully processed sequence for inference")
            return data
            
        except Exception as e:
            logger.error(f"Error in inference processing: {str(e)}")
            raise

# Amino acid to one-hot
def char_to_one_hot(char):
    mapping = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20  # Unknown or non-standard amino acid
    }
    return [mapping.get(char, 20)]  # Default to 'X' for unrecognized characters

# Example usage:
# inference = ProteinInference(sequence='ACDEFG')
# data = inference.process()