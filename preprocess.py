import pandas as pd


OUTPUT_TEXT = "data/ipl_docs.txt"
OUTPUT_PICKLE = "data/ipl_docs.pkl"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def row_to_text(row: pd.Series) -> str:
    # create a single-line summary of a delivery
    parts = []
    parts.append(f"match_id={row.get('match_id')}")
    parts.append(f"date={row.get('date')}")
    parts.append(f"{row.get('batting_team')} vs {row.get('bowling_team')}")
    parts.append(f"over={int(row.get('over',0))}.{int(row.get('ball',0))}")
    parts.append(f"batter={row.get('batter')}")
    parts.append(f"bowler={row.get('bowler')}")
    parts.append(f"runs={row.get('runs_total')}")
    wk = row.get('wicket_kind')
    if pd.notna(wk) and wk != '':
        parts.append(f"wicket={wk}")
    out = row.get('player_out')
    if pd.notna(out) and out != '':
        parts.append(f"player_out={out}")
    return " | ".join(parts)


def to_documents(df: pd.DataFrame) -> list[str]:
    texts = []
    for _, row in df.iterrows():
        texts.append(row_to_text(row))
    return texts


if __name__ == "__main__":
    import pickle
    import os

    input_path = "data/IPL.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    print("Loading CSV...")
    df = load_and_clean(input_path)
    print(f"rows = {len(df)}")

    print("Converting to text documents...")
    docs = to_documents(df)

    print(f"Writing {len(docs)} documents to {OUTPUT_TEXT}")
    os.makedirs(os.path.dirname(OUTPUT_TEXT), exist_ok=True)
    with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.replace("\n", " ") + "\n")

    print("Pickling document list for later retrieval use...")
    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(docs, f)

    print("Done")
