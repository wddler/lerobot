import pandas as pd

df = pd.read_parquet("/home/denis/.cache/huggingface/lerobot/denis/bolt_handover2/data/chunk-000/episode_000099.parquet")
print(df.columns)
print(df.head(3))

# Check if any cell contains the old directory name
# print(df.apply(lambda col: col.astype(str).str.contains("old_dir_name").any()))
