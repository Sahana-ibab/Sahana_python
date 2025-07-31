import pandas as pd

# Load parquet file
df = pd.read_parquet("de_train.parquet")

# View the first few rows
print(df[["SMILES"]])


num_unique_drugs = df["sm_name"].nunique()

print(f"Number of unique drugs used: {num_unique_drugs}")


unique_drugs = df["sm_name"].unique()
# Check if the positive controls are in the list
positive_controls = ["Dabrafenib", "Belinostat"]
for drug in positive_controls:
    if drug in unique_drugs:
        print(f"{drug} is counted in the dataset.")
    else:
        print(f"{drug} is NOT found in the dataset.")


unique_cell_types = df["cell_type"].unique()
print("Cell types are: ",unique_cell_types)