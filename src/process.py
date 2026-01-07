import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

def process_data():
    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw"
    processed_path = root / "data" / "processed"

    processed_path.mkdir(parents=True, exist_ok=True)

    # Ingestion: Get all match files from 2015 to 2024
    all_files = list(raw_path.glob("atp_matches_20*.csv"))

    if not all_files:
        print(f"No files found in {raw_path}")
        return 
    
    # read each CSV file
    df_list = [pd.read_csv(file) for file in all_files]

    # create full dataframe
    full_df = pd.concat(df_list, axis = 0, ignore_index = True)

    full_df['winner_rank'] = full_df['winner_rank'].fillna(999)
    full_df['loser_rank'] = full_df['loser_rank'].fillna(999)
    #print(full_df['winner_rank'])
    #print(full_df['loser_rank'])

    # random swapping to prevent data leakage
    mask = np.random.rand(len(full_df)) > 0.5

    # create and randomize player1 and player2 columns
    full_df['p1_name'] = np.where(mask, full_df['winner_name'], full_df['loser_name'])
    full_df['p2_name'] = np.where(mask, full_df['loser_name'], full_df['winner_name'])
    
    full_df['p1_rank'] = np.where(mask, full_df['winner_rank'], full_df['loser_rank'])
    full_df['p2_rank'] = np.where(mask, full_df['loser_rank'], full_df['winner_rank'])

    full_df['target'] = np.where(mask, 1, 0)
    #print(full_df.columns)
    #print(full_df)

    # save only necessary columns to processed df
    # cols_to_keep = [
    #     'tourney_date',
    #     'surface',
    #     'p1_name',
    #     'p2_name',
    #     'p1_rank',
    #     'p2_rank',
    #     'target'
    # ]
    # processed_df = full_df[cols_to_keep]
    processed_df = full_df.copy()

    # save to CSV file
    processed_df.to_csv(processed_path / 'atp_cleaned.csv', index = False)
    print(f"Merged {len(processed_df)} matches.")

    return processed_df

if __name__ == "__main__":
    process_data()






