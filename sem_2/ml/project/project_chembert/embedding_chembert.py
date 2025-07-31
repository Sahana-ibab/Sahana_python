# code reference:
# https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel


# Step 1: Loading the ChemBERTa model and tokenizer
def load_chemberta_model():
    # Loading the pre-trained tokenizer for the ChemBERTa model
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Loading the pre-trained ChemBERTa model
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Setting the model to evaluation mode (no training, just inference)
    model.eval()

    return tokenizer, model


# Step 2: Generating embeddings from a SMILES string
def get_chemberta_embedding(smiles, tokenizer, model):
    # Tokenizing the SMILES string to convert it into a format the model understands
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True)

    # Running the tokenized input through the ChemBERTa model to get the output
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Extracting the last hidden state and averaging over all tokens to get a single embedding for the molecule
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding


# Step 3: Loading the data from a Parquet file
def load_parquet_data(file_path):
    # Reading the Parquet file and loading it into a Pandas DataFrame
    df = pd.read_parquet(file_path)
    return df


# Step 4: Generating embeddings for all SMILES in the DataFrame
def generate_smiles_embeddings(df, tokenizer, model):
    embeddings = []

    # Iterating through each SMILES string in the DataFrame
    for smiles in df['SMILES']:
        # Generating an embedding for each SMILES string
        embedding = get_chemberta_embedding(smiles, tokenizer, model)
        embeddings.append(embedding)

    # Returning the list of embeddings as a NumPy array
    return np.array(embeddings)


# Step 5: Processing the data, generating embeddings, and saving the results
def process_and_save_embeddings(file_path, output_file):
    # Loading the data from the Parquet file
    df = load_parquet_data(file_path)

    # Loading the ChemBERTa model and tokenizer
    tokenizer, model = load_chemberta_model()

    # Generating embeddings for each SMILES in the DataFrame
    embeddings = generate_smiles_embeddings(df, tokenizer, model)

    # Converting the embeddings into a DataFrame with columns labeled as embedding_1, embedding_2, etc.
    embedding_df = pd.DataFrame(embeddings, columns=[f'embedding_{i + 1}' for i in range(embeddings.shape[1])])

    # Concatenating the original data with the embeddings DataFrame
    df = pd.concat([df[['cell_type', 'sm_name']], embedding_df, df[['A1BG']]], axis=1)

    # Saving the combined DataFrame (original data + embeddings) to a CSV file
    df.to_csv(output_file, index=False)

    print(f"Data with embeddings saved to {output_file}")


# Step 6: Running the code and executing the entire process
if __name__ == "__main__":
    input_file = "de_train.parquet"  # Specifying the input file path (Parquet format)
    output_file = "de_train_with_embeddings.csv"  # Specifying the output file path (CSV format)

    # Calling the function to process the data, generate embeddings, and save the results
    process_and_save_embeddings(input_file, output_file)
