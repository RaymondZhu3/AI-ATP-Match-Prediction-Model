import pandas as pd
import numpy as np
import math
from collections import defaultdict

def update_elo(rating1, rating2, result, k):
    expected = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    change = k * (result - expected)
    return rating1 + change, rating2 - change

def compute_elo_features(df, k_factor=32):
    # sort by date
    df = df.sort_values('tourney_date').reset_index(drop=True)

    # initalize elo ratings to 1500
    elo_general = defaultdict(lambda: 1500)
    elo_surface = {surface: defaultdict(lambda: 1500) for surface in df['surface'].unique()}

    p1_gen_elo, p2_gen_elo = [], []
    p1_surf_elo, p2_surf_elo = [], []

    # record and update elos for every match
    for row in df.itertuples():
        surface = row.surface

        p1_gen_rating = elo_general[row.p1_name]
        p2_gen_rating = elo_general[row.p2_name]
        p1_surf_rating = elo_surface[surface][row.p1_name]
        p2_surf_rating = elo_surface[surface][row.p2_name]

        # record before ratings
        p1_gen_elo.append(p1_gen_rating)
        p2_gen_elo.append(p2_gen_rating)
        p1_surf_elo.append(p1_surf_rating)
        p2_surf_elo.append(p2_surf_rating)

        # update elos based on formula
        actual = 1 if row.target == 1 else 0
        new_p1_gen, new_p2_gen = update_elo(p1_gen_rating, p2_gen_rating, actual, k_factor)
        elo_general[row.p1_name] = new_p1_gen
        elo_general[row.p2_name] = new_p2_gen
        new_p1_surf, new_p2_surf = update_elo(p1_surf_rating, p2_surf_rating, actual, k_factor)
        elo_surface[surface][row.p1_name] = new_p1_surf
        elo_surface[surface][row.p2_name] = new_p2_surf
    
    df['p1_gen_elo'], df['p2_gen_elo'] = p1_gen_elo, p2_gen_elo
    df['p1_surf_elo'], df['p2_surf_elo'] = p1_surf_elo, p2_surf_elo

    # one hot encode the surface
    df = pd.get_dummies(df, columns=['surface'], prefix='surf')

    # print(dict(elo_dict))
    # # Sort players by their Elo rating and print the top 10
    # sorted_elo = sorted(elo_dict.items(), key=lambda x: x[1], reverse=True)
    # for player, rating in sorted_elo[:10]:
    #     print(f"{player}: {rating:.2f}")

    # # Check if Alcaraz's Elo is updating correctly
    # print(df[df['p1_name'] == 'Carlos Alcaraz'][['tourney_date', 'p1_name', 'p1_elo', 'target']].head(5))
    return df


def add_momentum_features(df, window_size=5):
    # Sort matches chronologically
    df = df.sort_values('tourney_date')
    
    # calculate rolling win rate for Player 1
    df[f'p1_win_rate_last_{window_size}'] = df.groupby('p1_id')['target'].transform(
        lambda x: x.shift(1).rolling(window=window_size).mean()
    )

    # calculate rolling win rate for Player 2
    df[f'p2_win_rate_last_{window_size}'] = df.groupby('p2_id')['target'].transform(
        lambda x: (1 - x).shift(1).rolling(window=window_size).mean()
    )
    return df

if __name__ == "__main__":
    from pathlib import Path
    base_path = Path(__file__).resolve().parent.parent
    test_path = base_path / "data" / "processed" / "atp_cleaned.csv"
    output_path = base_path / "data" / "processed" / "atp_with_elo.csv"

    if test_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(test_path)
        df_with_elo = compute_elo_features(df)
        df_with_elo.to_csv(output_path, index=False)
    else:
        print(f"Error: Could not find file at {test_path}")
