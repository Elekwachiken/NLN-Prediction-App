import pandas as pd


# Updated validate_file_type to normalize columns
def normalize_columns(cols):
    seen = {}
    normalized = []
    for col in cols:
        norm = col.strip().lower().replace(" ", "").replace('@timestamp', 'timestamp')
        # Ensure uniqueness by appending suffixes if duplicated
        if norm in seen:
            seen[norm] += 1
            norm += f"_{seen[norm]}"
        else:
            seen[norm] = 0
        normalized.append(norm)
    return normalized


def validate_file_type(df):
    normalized_cols = normalize_columns(df.columns)
    required_game = normalize_columns(['timestamp', 'PLAYER ID', 'price', 'prize', 'ticketStatus', 'accessChannel', 'Game'])
    required_wallet = normalize_columns(['timestamp', 'channel', 'player id', 'amount', 'Action'])

    if all(col in normalized_cols for col in required_game):
        return 'game'
    elif all(col in normalized_cols for col in required_wallet):
        return 'wallet'
    else:
        raise ValueError("Uploaded file does not match required game or wallet column structure.")


def coalesce_duplicate_columns(df, base_col_name):
    """Coalesce duplicated columns like playerid and playerid_1 into one."""
    similar_cols = [col for col in df.columns if col.startswith(base_col_name)]
    if not similar_cols:
        return df
    # Combine them left to right
    df[base_col_name] = df[similar_cols].bfill(axis=1).iloc[:, 0]
    df.drop(columns=[col for col in similar_cols if col != base_col_name], inplace=True)
    return df


def clean_game_df(df):
    df = df.copy()
    # Normalize column names early
    df.columns = normalize_columns(df.columns)

    # Coalesce variations
    df = coalesce_duplicate_columns(df, 'playerid')
    df = coalesce_duplicate_columns(df, 'timestamp')

    # Rename relevant columns to standardized names
    rename_map = {
        'playerid': 'player_id',
        'timestamp': 'timestamp',
        'price': 'stake',
        'prize': 'prize',
        'ticketstatus': 'ticket_status',
        'accesschannel': 'access_channel',
        'game': 'game'
    }
    df.rename(columns=rename_map, inplace=True)

    if 'player_id' not in df.columns:
        raise ValueError("Missing required column: player_id")

    # print("DEBUG Columns:", df.columns.tolist())
    # print("DEBUG Types:", df.dtypes)
    df['player_id'] = df['player_id'].astype(str).str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['stake'] = pd.to_numeric(df['stake'], errors='coerce')
    df['prize'] = pd.to_numeric(df['prize'], errors='coerce')
    # df['game'] = df['game'].apply(clean_game_name)
    df['game'] = df['game'].astype(str).apply(lambda g: str(g).strip().upper())

    df.dropna(inplace=True)
    return df


def clean_wallet_df(df):
    df = df.copy()
    # Normalize column names early
    df.columns = normalize_columns(df.columns)

    df = coalesce_duplicate_columns(df, 'playerid')
    df = coalesce_duplicate_columns(df, 'timestamp')

    # Rename relevant columns to standardized names
    rename_map = {
        'playerid': 'player_id',
        'timestamp': 'timestamp',
        'channel': 'channel',
        'amount': 'amount',
        'action': 'action'
    }
    df.rename(columns=rename_map, inplace=True)

    if 'player_id' not in df.columns:
        raise ValueError("Missing required column: player_id")

    # print("DEBUG Columns:", df.columns.tolist())
    # print("DEBUG Types:", df.dtypes)
    df['player_id'] = df['player_id'].astype(str).str.strip()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df.dropna(inplace=True)
    return df

# def clean_game_name(game):
#     game = str(game).strip()
#     if 'scratch' in game.lower():
#         prefix = game.lower().replace('scratch', '').strip().upper()
#         return f"{prefix}Scratch"
#     return game.strip().upper()
