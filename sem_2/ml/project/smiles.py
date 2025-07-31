import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_all_descriptors(smiles):
    """Compute all RDKit molecular descriptors for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES
        return None
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    return {name: getattr(Descriptors, name)(mol) for name in descriptor_names}
#
# # Load dataset
# df = pd.read_parquet("de_train.parquet")
#
# # Compute descriptors for each SMILES
# descriptor_data = df["SMILES"].apply(get_all_descriptors)
#
# # Convert list of dictionaries into a DataFrame
# descriptor_df = pd.DataFrame(descriptor_data.tolist())
#
# # Combine original dataset with descriptors
# new_df = pd.concat([df, descriptor_df], axis=1)
#
# # # Save the new dataset
# # new_df.to_csv("SMILES.csv", index=False)
# #
# # print("Dataset with molecular descriptors saved successfully!")
#
# # sm = pd.read_csv("SMILES.csv")
# # print("....",sm)
#
# import pandas as pd
#
# # Load dataset with descriptors
# df = pd.read_csv("SMILES.csv")
# print(df.head())
# # Drop columns where all values are zero
# df_filtered = df.loc[:, (df != 0).any(axis=0)]
#
# # Save the cleaned dataset
# df_filtered.to_csv("SMILES_filtered.csv", index=False)
# print(df_filtered.columns)
# print("Filtered dataset saved successfully!")


# import pandas as pd
#
# # Load the original and filtered datasets
# df_original = pd.read_parquet("de_train.parquet")
# df_filtered = pd.read_csv("SMILES_filtered.csv")
#
# # Get the column names as sets
# original_cols = set(df_original.columns)
# filtered_cols = set(df_filtered.columns)
#
# # Find columns that are in filtered but not in original
# newly_added_cols = list(filtered_cols - original_cols)
#
# # Get their index positions in the filtered DataFrame
# indices_of_new_cols = [df_filtered.columns.get_loc(col) for col in newly_added_cols]
# print(len(indices_of_new_cols))
# print("Newly added columns:", newly_added_cols)
# print("Indices of newly added columns:", indices_of_new_cols)


# # Load dataset
# df = pd.read_parquet("de_train.parquet")
#
# # Compute descriptors for each SMILES
# descriptor_data = df["SMILES"].apply(get_all_descriptors)
#
# # Convert list of dictionaries into a DataFrame
# descriptor_df = pd.DataFrame(descriptor_data.tolist())
#
# # Combine original dataset with descriptors
# new_df = pd.concat([descriptor_df, df.iloc[:, [5]]], axis=1)
#
# # # Save the new dataset
# new_df.to_csv("SMILES_only.csv", index=False)
#
# print("Dataset with molecular descriptors saved successfully!")


df = pd.read_csv("SMILES_only.csv")
df_filtered = df.loc[:, (df != 0).any(axis=0)]

# Save the cleaned dataset
df_filtered.to_csv("SMILES_only_filtered.csv", index=False)
print(df_filtered.columns)
print("Filtered dataset saved successfully!")


