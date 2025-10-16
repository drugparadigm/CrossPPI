import pandas as pd
import torch
import esm
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Loaded ESM-2 model on device: {device}")

def generate_contact_map(sequence, complex_id, threshold=0.5, max_length=999):
    """Generate a contact map for a protein sequence using ESM-2 and apply a threshold."""
    # Truncate sequence to max_length (matching Protein_dataset)
    original_length = len(sequence)
    sequence = sequence[:max_length] if len(sequence) > max_length else sequence
    if len(sequence) != original_length:
        logger.warning(f"Truncated sequence for Complex ID {complex_id} from {original_length} to {len(sequence)} residues")

    # Validate sequence
    if not isinstance(sequence, str) or not sequence:
        logger.error(f"Invalid sequence for Complex ID {complex_id}")
        raise ValueError(f"Invalid sequence for Complex ID {complex_id}")

    # Tokenize the sequence
    batch_converter = alphabet.get_batch_converter()
    data = [("", sequence)]  # ESM expects a tuple with (name, sequence)
    try:
        _, _, tokens = batch_converter(data)
    except Exception as e:
        logger.error(f"Error tokenizing sequence for Complex ID {complex_id}: {e}")
        raise

    tokens = tokens.to(device)

    # Generate contact map
    with torch.no_grad():
        try:
            results = model(tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].cpu().numpy()  # Shape: [len(sequence), len(sequence)]
        except Exception as e:
            logger.error(f"Error generating contact map for Complex ID {complex_id}: {e}")
            raise

    # Validate contact map dimensions
    if contact_map.shape != (len(sequence), len(sequence)):
        logger.error(f"Contact map shape {contact_map.shape} does not match sequence length {len(sequence)} for Complex ID {complex_id}")
        raise ValueError(f"Contact map shape mismatch for Complex ID {complex_id}")

    # Apply threshold to binarize the contact map
    binary_contact_map = (contact_map > threshold).astype(np.int8)
    logger.info(f"Generated contact map for Complex ID {complex_id}, shape: {binary_contact_map.shape}, max index: {np.argwhere(binary_contact_map == 1).max() if np.argwhere(binary_contact_map == 1).size > 0 else 'No edges'}")
    return binary_contact_map

def save_contact_map(contact_map, complex_id, output_dir="contact_maps_ligands"):
    """Save the contact map to a text file named after the complex ID."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{complex_id}.txt")
    try:
        np.savetxt(output_file, contact_map, fmt="%d")
        logger.info(f"Saved contact map for Complex ID {complex_id} to {output_file}")
    except Exception as e:
        logger.error(f"Error saving contact map for Complex ID {complex_id}: {e}")
        raise

def process_dataframe(df, sequence_col="Receptor Chains", id_col="Complex ID", threshold=0.5, max_length=999):
    """Iterate through DataFrame and generate contact maps for each sequence, displaying rows left."""
    total_rows = len(df)
    skipped_entries = []

    for index, row in df.iterrows():
        rows_left = total_rows - (index + 1)
        complex_id = row[id_col]
        logger.info(f"Processing row {index + 1}/{total_rows} (Complex ID: {complex_id}), {rows_left} rows left")

        sequence = row[sequence_col]

        # Validate sequence
        if not isinstance(sequence, str) or not sequence:
            logger.warning(f"Skipping invalid sequence for Complex ID {complex_id}")
            skipped_entries.append(complex_id)
            continue

        # Generate and save contact map
        try:
            contact_map = generate_contact_map(sequence, complex_id, threshold, max_length)
            save_contact_map(contact_map, complex_id)
        except Exception as e:
            logger.warning(f"Skipping Complex ID {complex_id} due to error: {e}")
            skipped_entries.append(complex_id)
            continue

    if skipped_entries:
        logger.info(f"Skipped {len(skipped_entries)} entries: {skipped_entries}")
    else:
        logger.info("All entries processed successfully")

# Example usage
if __name__ == "__main__":
    # Load DataFrame from Excel file
    excel_file = "data/PPB-Affinity-Modified(pkd).xlsx"  # Match path from Protein_dataset
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        logger.error(f"Excel file '{excel_file}' not found")
        exit(1)
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        exit(1)

    # Verify required columns
    required_columns = ["Complex ID", "Receptor Chains"]
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Excel file must contain columns: {required_columns}")
        exit(1)

    # Process the DataFrame
    process_dataframe(df, sequence_col="Ligand Chains", id_col="Complex ID", threshold=0.5, max_length=999)