import pandas as pd

def build_player_features(game_df, wallet_df):
    # Debugging: Initial state of input dataframes
    print(f"build_player_features: game_df shape: {game_df.shape}, wallet_df shape: {wallet_df.shape}")


    # Ensure dataframes are not empty before processing
    if game_df.empty:
        print("Warning: game_df is empty in build_player_features.")
        # Return an empty DataFrame with expected columns if game_df is crucial for merge
        return pd.DataFrame(columns=['player_id', 'first_play_date', 'last_play_date',
                                     'total_days_active', 'num_plays', 'total_stake',
                                     'total_prize', 'avg_stake', 'avg_prize',
                                     'distinct_games_played', 'most_played_game',
                                     'most_played_game_count', 'most_used_channel', 'channel_count',
                                     'total_amount_deposit', 'txn_count_deposit',
                                     'total_amount_withdrawal', 'txn_count_withdrawal',
                                     'days_since_last_play', 'churned', 'Tenure_Days',
                                     'net_revenue', 'loss_ratio', 'win_loss_ratio',
                                     'net_revenue_per_play', 'ltv', 'churn_likelihood_score',
                                     'likely_to_churn', 'reactivated', 
                                     'R_score', 'F_score', 'M_score', 'RFM_Segment']) # New RFM columns

    # Game Summary Features
    game_summary = game_df.groupby('player_id').agg(
        first_play_date=('timestamp', 'min'),
        last_play_date=('timestamp', 'max'),
        total_days_active=('timestamp', lambda x: x.dt.date.nunique()),
        num_plays=('stake', 'count'),
        total_stake=('stake', 'sum'),
        total_prize=('prize', 'sum'),
        avg_stake=('stake', 'mean'),
        avg_prize=('prize', 'mean'),
        distinct_games_played=('game', 'nunique'),
        # Ensure idxmax() handles empty series gracefully if a group is too small
        most_played_game=('game', lambda x: x.value_counts().idxmax() if not x.empty else None),
        most_played_game_count=('game', lambda x: x.value_counts().max() if not x.empty else 0),
        most_used_channel=('access_channel', lambda x: x.value_counts().idxmax() if not x.empty else None),
        channel_count=('access_channel', 'nunique')
    ).reset_index()
    print(f"game_summary shape: {game_summary.shape}, columns: {game_summary.columns.tolist()}")

    # Wallet Summary Features
    # Handle cases where wallet_df might be empty or missing 'action' column
    if wallet_df.empty or 'action' not in wallet_df.columns:
        print("Warning: wallet_df is empty or missing 'action' column in build_player_features. Skipping wallet summary.")
        wallet_summary = pd.DataFrame(columns=['player_id', 'total_amount_deposit', 'txn_count_deposit',
                                               'total_amount_withdrawal', 'txn_count_withdrawal'])
    else:
        wallet_summary = wallet_df.groupby(['player_id', 'action']).agg(
            total_amount=('amount', 'sum'),
            txn_count=('amount', 'count')
        ).unstack(fill_value=0)
        wallet_summary.columns = ['_'.join(col).lower() for col in wallet_summary.columns]
        wallet_summary = wallet_summary.reset_index()
        print(f"wallet_summary shape: {wallet_summary.shape}, columns: {wallet_summary.columns.tolist()}")

    # Merge game and wallet summaries
    player_features = pd.merge(game_summary, wallet_summary, on='player_id', how='left')
    print(f"player_features after merge shape: {player_features.shape}, columns: {player_features.columns.tolist()}")

    # Fill NaN values for wallet features that might not exist for all players
    # This is crucial if a player only has game data but no wallet transactions
    wallet_cols = [col for col in wallet_summary.columns if col != 'player_id']
    for col in wallet_cols:
        if col not in player_features.columns: # Add column if it doesn't exist after merge
            player_features[col] = 0
        player_features[col] = player_features[col].fillna(0)
    print(f"player_features after wallet NaN fill shape: {player_features.shape}")


    # Time-based Features
    today = game_df['timestamp'].max() # Use the latest timestamp from the game data as 'today'
    player_features['days_since_last_play'] = (today - player_features['last_play_date']).dt.days
    player_features['churned'] = (player_features['days_since_last_play'] > 7).astype(int) # Example threshold
    player_features['Tenure_Days'] = (player_features['last_play_date'] - player_features['first_play_date']).dt.days.fillna(0)

    # Financial Features
    player_features['net_revenue'] = player_features['total_stake'] - player_features['total_prize']
    # Handle division by zero for ratios
    player_features['loss_ratio'] = player_features['net_revenue'] / player_features['total_stake'].replace(0, 1)
    player_features['win_loss_ratio'] = player_features['total_prize'] / player_features['total_stake'].replace(0, 1)
    player_features['net_revenue_per_play'] = player_features['net_revenue'] / player_features['num_plays'].replace(0, 1)
    player_features['ltv'] = player_features['net_revenue'] / player_features['total_days_active'].replace(0, 1)

    # Churn Likelihood Score (custom, not used for model input but for display)
    max_days_since_last_play = player_features['days_since_last_play'].max()
    if max_days_since_last_play > 0:
        player_features['churn_likelihood_score'] = (
            player_features['days_since_last_play'].fillna(0) / max_days_since_last_play
        )
    else:
        player_features['churn_likelihood_score'] = 0 # All players are equally likely if no max days
    player_features['likely_to_churn'] = (player_features['churn_likelihood_score'] > 0.6).astype(int)


    # NEW: Reactivation after win
    # Ensure game_df is sorted for accurate tracking
    game_df_sorted = game_df.sort_values(['player_id', 'timestamp']).copy()

    # Find first win date per player
    first_win = game_df_sorted[game_df_sorted['prize'] > 0].groupby('player_id')['timestamp'].min().reset_index()
    first_win.columns = ['player_id', 'first_win_date']

    # Merge back to check for subsequent plays
    # Use original game_df for merging to avoid issues with already grouped data
    player_activity_after_win = pd.merge(game_df_sorted, first_win, on='player_id', how='left')

    # Check if a player had any activity (any timestamp) after their first win date
    # This correctly identifies if they reactivated by playing again
    reactivation_summary = player_activity_after_win[
        player_activity_after_win['timestamp'] > player_activity_after_win['first_win_date']
    ].groupby('player_id').size().reset_index(name='plays_after_win_count')

    # Determine if reactivated (had at least one play after first win)
    reactivation_summary['reactivated'] = reactivation_summary['plays_after_win_count'] > 0

    # Merge 'reactivated' status into player_features
    # Use a left merge to keep all players in player_features, filling False for those who never won or didn't reactivate
    player_features = pd.merge(player_features, reactivation_summary[['player_id', 'reactivated']], on='player_id', how='left')
    player_features['reactivated'] = player_features['reactivated'].fillna(False) # Fill NaN for players who never had a win


    #  # NEW: RFM Segmentation
    # # Calculate R, F, M scores (1-5 scale)
    # # Recency: Lower days_since_last_play is better (higher score)
    # # Handle cases where days_since_last_play might be uniform or missing for qcut
    # if player_features['days_since_last_play'].nunique() >= 5:
    #     player_features['R_score'] = pd.qcut(player_features['days_since_last_play'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    # else:
    #     # player_features['R_score'] = 1 # Assign lowest score if no variation
    #     player_features['R_score'] = 3  # Default midpoint score

    # # Frequency: Higher num_plays is better (higher score)
    # if player_features['num_plays'].nunique() >= 5:
    #     player_features['F_score'] = pd.qcut(player_features['num_plays'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    # else:
    #     # player_features['F_score'] = 1 # Assign lowest score if no variation
    #     player_features['F_score'] = 3  # Default midpoint score

    # # Monetary: Higher net_revenue is better (higher score)
    # # Implement the recommended strategy for monetary score with zero/negative values
    # if player_features['net_revenue'].nunique() >= 5:
    #     monetary_min = player_features['net_revenue'].min()
    #     if monetary_min <= 0:
    #         player_features['Monetary_Adjusted'] = player_features['net_revenue'] + abs(monetary_min) + 1
    #         # Use labels=False to get 0-4 range, then add 1 to make it 1-5
    #         player_features['M_score'] = pd.qcut(player_features['Monetary_Adjusted'], q=5, labels=False, duplicates='drop').astype(int) + 1
    #     else:
    #         player_features['M_score'] = pd.qcut(player_features['net_revenue'], q=5, labels=False, duplicates='drop').astype(int) + 1
    # else:
    #     # player_features['M_score'] = 1 # Assign lowest score if no variation
    #     player_features['M_score'] = 3  # Default midpoint score

    # Recency (R)
    if player_features['days_since_last_play'].nunique() >= 2:
        try:
            _, bins = pd.qcut(player_features['days_since_last_play'], q=5, retbins=True, duplicates='drop')
            num_bins = len(bins) - 1
            if num_bins >= 2:
                r_labels = list(range(num_bins, 0, -1))  # e.g., [5, 4, 3, 2, 1] if num_bins=5
                player_features['R_score'] = pd.qcut(player_features['days_since_last_play'], q=num_bins, labels=r_labels).astype(int)
            else:
                player_features['R_score'] = 3
        except Exception:
            player_features['R_score'] = 3
    else:
        player_features['R_score'] = 3

    # Frequency (F)
    if player_features['num_plays'].nunique() >= 2:
        try:
            _, bins = pd.qcut(player_features['num_plays'], q=5, retbins=True, duplicates='drop')
            num_bins = len(bins) - 1
            if num_bins >= 2:
                f_labels = list(range(1, num_bins + 1))  # e.g., [1,2,3,4,5]
                player_features['F_score'] = pd.qcut(player_features['num_plays'], q=num_bins, labels=f_labels).astype(int)
            else:
                player_features['F_score'] = 3
        except Exception:
            player_features['F_score'] = 3
    else:
        player_features['F_score'] = 3

    # Monetary (M)
    if player_features['net_revenue'].nunique() >= 2:
        try:
            monetary_min = player_features['net_revenue'].min()
            if monetary_min <= 0:
                player_features['Monetary_Adjusted'] = player_features['net_revenue'] + abs(monetary_min) + 1
                _, bins = pd.qcut(player_features['Monetary_Adjusted'], q=5, retbins=True, duplicates='drop')
                num_bins = len(bins) - 1
                if num_bins >= 2:
                    player_features['M_score'] = pd.qcut(player_features['Monetary_Adjusted'], q=num_bins, labels=False).astype(int) + 1
                else:
                    player_features['M_score'] = 3
            else:
                _, bins = pd.qcut(player_features['net_revenue'], q=5, retbins=True, duplicates='drop')
                num_bins = len(bins) - 1
                if num_bins >= 2:
                    player_features['M_score'] = pd.qcut(player_features['net_revenue'], q=num_bins, labels=False).astype(int) + 1
                else:
                    player_features['M_score'] = 3
        except Exception:
            player_features['M_score'] = 3
    else:
        player_features['M_score'] = 3


    # Convert scores to int
    player_features['R_score'] = player_features['R_score'].astype(int)
    player_features['F_score'] = player_features['F_score'].astype(int)
    player_features['M_score'] = player_features['M_score'].astype(int)

    # Define RFM Segments based on scores
    def rfm_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 4 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        elif r == 5 and f == 1 and m == 1:
            return 'New Customers'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        else:
            return 'Others' # Catch-all for less common combinations

    player_features['RFM_Segment'] = player_features.apply(rfm_segment, axis=1)

    # Fill any remaining NaN values that might result from calculations (e.g., for new players)
    player_features = player_features.fillna(0)
    print(f"player_features before return shape: {player_features.shape}, columns: {player_features.columns.tolist()}")
    print(f"player_features head before return:\n{player_features.head()}")

    return player_features