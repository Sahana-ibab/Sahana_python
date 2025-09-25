import pandas as pd

file = "landmark_genes.csv"
file2= "target_genes.csv"
df = pd.read_csv(file, sep=",")  # or sep="," if it's CSV
print("landmark gene: ",df.shape)
df2 = pd.read_csv(file2, sep=",")
print("target gene: ",df2.shape)
#
# #!/usr/bin/env python3
# import pandas as pd
#
# # Input and output file names
# input_file = "landmark_genes.txt"
# output_file = "landmark_genes.csv"
#
# # Read the text file
# # If it's tab-delimited, use sep="\t"
# # If it's space-delimited, use delim_whitespace=True
# df = pd.read_csv(input_file, sep="\t")  # change sep if needed
#
# # Save as CSV
# df.to_csv(output_file, index=False)
#
# print(f"Converted {input_file} to {output_file} with shape {df.shape}")
#
