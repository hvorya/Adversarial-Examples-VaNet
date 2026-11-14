import pandas as pd

input_file = r"D:\Greece\vanet-federated-nextgen-main\results\all\process\test.parquet"
output_file = r"D:\Greece\vanet-federated-nextgen-main\results\all\process\test.csv"

df = pd.read_parquet(input_file)
df.to_csv(output_file, index=False)

print("Converted successfully:", df.shape)