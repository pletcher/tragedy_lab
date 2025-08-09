import json

import polars as pl

from .normalization import to_conllu


def get_lines_from_dataframe(df: pl.DataFrame, urn: str, lines: list[int]):
    rows = (
        df.filter(
            pl.col("urn").str.starts_with(urn)
            & pl.col("n").ge(lines[0])
            & pl.col("n").le(lines[1])
        ).select(pl.col("speaker"), pl.col("text"))
    ).rows()

    speaker = rows[0][0]
    text = [r[1] for r in rows]

    return (speaker, text)


def write_messenger_speeches_to_docs():
    tragedy_df = pl.read_parquet("./tragedy.parquet")

    with open("./messenger_speeches_updated.json") as f:
        speeches = json.load(f)

    for speech in speeches:
        line_numbers = speech["speeches"]

        for pair in line_numbers:
            speaker, text = get_lines_from_dataframe(tragedy_df, speech["urn"], pair)

            filename = f"./messenger_corpus/{speech['title']}_{speaker}_{pair[0]}-{pair[1]}.txt"

            with open(filename, "w") as f:
                f.write("\n".join(text))


def write_messenger_speeches_to_conllu():
    to_conllu("./messenger_corpus", "./messenger_conllu")
