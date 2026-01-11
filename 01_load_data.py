import h5py  # Import h5py library for exploring HDF5 file structure
import pandas as pd  # Import pandas library for working with CSV files
import scanpy as sc  # Import scanpy library for reading h5ad files
import os  # Import os for file path operations
import numpy as np  # Import numpy for array operations (if needed)

# Define the file paths
h5_file_path = "competition_support_set/competition_train.h5"  # Path to the H5 file (h5ad-compatible)
csv_file_path = "competition_support_set/gene_names.csv"  # Path to the CSV file

# Check if files exist
if not os.path.exists(h5_file_path):
    print(f"Error: {h5_file_path} not found!")
    exit(1)

if not os.path.exists(csv_file_path):
    print(f"Error: {csv_file_path} not found!")
    exit(1)

# Function to recursively list all keys in an HDF5 file
def list_h5_keys(file, prefix=""):
    """Recursively list all keys (datasets and groups) in an HDF5 file."""
    keys = []
    # Iterate through all items in the current group
    for key in file.keys():
        full_path = f"{prefix}/{key}" if prefix else key
        # Check if the item is a group (like a folder) or a dataset (actual data)
        if isinstance(file[key], h5py.Group):
            # If it's a group, recursively explore it
            keys.append(full_path + " (group)")
            keys.extend(list_h5_keys(file[key], full_path))
        else:
            # If it's a dataset, add it with its shape information
            shape = file[key].shape
            keys.append(f"{full_path} (dataset, shape: {shape})")
    return keys

print("=" * 60)
print("Exploring the H5 file structure...")
print("=" * 60)

# Open the H5 file in read mode to explore its structure
with h5py.File(h5_file_path, 'r') as f:
    # List all keys in the H5 file
    print("\nAll keys inside the H5 file:")
    print("-" * 60)
    all_keys = list_h5_keys(f)
    for key in all_keys:
        print(f"  {key}")
    
    print("\n" + "=" * 60)

# Load the gene names from CSV file using pandas
print("\nLoading gene names from CSV...")
print("-" * 60)
gene_names_df = pd.read_csv(csv_file_path)  # Read the CSV file into a pandas DataFrame
print(f"Loaded gene names CSV with shape: {gene_names_df.shape}")
print(f"Columns in CSV: {list(gene_names_df.columns)}")
print(f"\nFirst few rows of gene names:")
print(gene_names_df.head())

print("\n" + "=" * 60)

# Load the H5 file using scanpy.read_h5ad (even though it ends in .h5, it's h5ad-compatible)
print("\nLoading H5 file using scanpy.read_h5ad()...")
print("-" * 60)
adata = sc.read_h5ad(h5_file_path)  # Read the h5ad-compatible H5 file

# Print the number of cells (observations) and number of genes (variables)
print(f"\nNumber of cells (n_obs): {adata.n_obs}")  # Print number of cells
print(f"Number of genes (n_vars): {adata.n_vars}")  # Print number of genes

print("\n" + "=" * 60)

# Access the target_gene column in the observation metadata
print("\nExtracting target genes from adata.obs['target_gene']...")
print("-" * 60)

# Check if 'target_gene' column exists
if 'target_gene' in adata.obs.columns:
    # Get the target_gene column
    target_gene_col = adata.obs['target_gene']  # Extract the target_gene column
    
    # Check if it's stored as categorical with codes and categories
    # In h5ad format, categoricals are stored with codes (integers) and categories (strings)
    if pd.api.types.is_categorical_dtype(target_gene_col):  # Check if it's a pandas Categorical dtype
        # It's stored as categorical, decode codes to get actual gene names
        # Use astype('str') to convert from categorical codes to string gene names
        target_genes_decoded = target_gene_col.astype('str')  # Convert categorical to string (decodes codes to categories)
        unique_target_genes = target_gene_col.cat.categories.tolist()  # Get unique categories (gene names) directly
        print("Found target_gene as categorical. Decoded to actual gene names.")
        print(f"  Categorical has {len(target_gene_col.cat.categories)} unique categories")
    else:
        # It's already in readable format (not categorical)
        unique_target_genes = pd.unique(target_gene_col).tolist()  # Get unique gene names directly
        print("Found target_gene as regular column.")
    
    # Print the unique gene names
    print(f"\nNumber of unique target genes: {len(unique_target_genes)}")
    print(f"\nUnique target gene names:")
    for i, gene in enumerate(sorted(unique_target_genes), 1):  # Sort for easier reading
        print(f"  {i}. {gene}")
else:
    print("Warning: 'target_gene' column not found in adata.obs")
    print(f"Available columns in adata.obs: {list(adata.obs.columns)}")

print("\n" + "=" * 60)
print("Data loading complete!")
print("=" * 60)
