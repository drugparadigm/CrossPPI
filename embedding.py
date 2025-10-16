import pandas as pd
import torch
import esm
import numpy as np
import os
from datetime import datetime

# Load the ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
device = "cuda"
model.to(device)

def generate_embeddings(sequence, max_length=1000):
    """Generate per-residue embeddings for a protein sequence using ESM-2."""
    # Check sequence length
    if len(sequence) > max_length:
        print(f"Sequence too long ({len(sequence)} residues). Skipping.")
        return None, len(sequence)
    
    # Tokenize the sequence
    batch_converter = alphabet.get_batch_converter()
    data = [("", sequence)]  # ESM expects a tuple with (name, sequence)
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    
    # Generate embeddings
    try:
        with torch.no_grad():
            results = model(tokens, repr_layers=[33], return_contacts=False)
            embeddings = results["representations"][33][0].cpu().numpy()  # Per-residue embeddings
        return embeddings, len(sequence)
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        return None, len(sequence)
    finally:
        # Clear memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def save_embeddings(embeddings, complex_id, output_dir="embeddings_receptor"):
    """Save the embeddings to a .npy file named after the complex ID."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"{complex_id}.npy")
    np.save(output_file, embeddings)
    print(f"Saved embeddings for {complex_id} to {output_file}")

def log_failed_sequence(complex_id, sequence_length, error, log_file="failed_sequences.log"):
    """Log details of failed sequences to a file."""
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Complex ID: {complex_id}, Length: {sequence_length}, Error: {str(error)}\n")

def process_dataframe(df, sequence_col="sequence", id_col="complex_id", max_length=1000):
    """Iterate through DataFrame and generate embeddings for each sequence, displaying rows left."""
    total_rows = len(df)
    for index, row in df.iterrows():
        rows_left = total_rows - (index + 1)
        print(f"Processing row {index + 1}/{total_rows}, {rows_left} rows left")
        
        complex_id = row[id_col]
        sequence = row[sequence_col]
        
        # Validate sequence
        if not isinstance(sequence, str) or not sequence:
            print(f"Skipping invalid sequence for {complex_id}")
            log_failed_sequence(complex_id, 0, "Invalid or empty sequence")
            continue
        
        # Generate embeddings
        try:
            embeddings, seq_length = generate_embeddings(sequence, max_length)
            if embeddings is not None:
                save_embeddings(embeddings, complex_id)
            else:
                print(f"Failed to process {complex_id}: Sequence too long or memory error")
                log_failed_sequence(complex_id, seq_length, "Sequence too long or memory error")
        except Exception as e:
            print(f"Error processing {complex_id}: {e}")
            log_failed_sequence(complex_id, len(sequence), str(e))

# Example usage
if __name__ == "__main__":
    # Load DataFrame from Excel file
    excel_file = "PPB-Affinity-Modified.xlsx"  # Excel file path
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"Error: Excel file '{excel_file}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(1)
    
    # Verify required columns
    if "Complex ID" not in df.columns or "Receptor Chains" not in df.columns:
        print("Error: Excel file must contain 'Complex ID' and 'Ligand Chains' columns.")
        exit(1)
    
    # Process the DataFrame
    process_dataframe(df, sequence_col="Receptor Chains", id_col="Complex ID", max_length=1000)