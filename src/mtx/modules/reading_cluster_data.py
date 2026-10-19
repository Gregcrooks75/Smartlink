import pandas as pd
from pathlib import Path

def saving_by_chunks(input_path, division, chunk_size=2_000_000):

    output_dir = Path(
        r"C:\Users\u355394\OneDrive - IBERDROLA S.A"
        r"\Documents - Asset Data Strategy"
        r"\Asset Data Quality"
        r"\Meter to Transformer Analysis"
        fr"\pickles\{division}\Cluster Data"
    )

    # If pickle files already exist → read & concat
    if output_dir.exists():
        pickle_files = sorted(output_dir.glob("*.pkl"))

        if pickle_files:
            print(f"Found {len(pickle_files)} existing Cluster pickle files for {division}, reading and concatenating them into one DataFrame")
            dfs = [pd.read_pickle(pkl) for pkl in pickle_files]
            return pd.concat(dfs, ignore_index=True)

    # Otherwise → create directory and write chunks
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
        output_path = output_dir / f"{division}_part_{i + 1}.pkl"
        chunk.to_pickle(output_path)
        written_files.append(output_path)
        print(f"Saved {output_path} with shape {chunk.shape}")

    # Read back and concat
    dfs = [pd.read_pickle(pkl) for pkl in written_files]
    return pd.concat(dfs, ignore_index=True)