import json
import time

from typing import Literal

import polars as pl
import requests


class DICESClient:
    base_url = "http://dices.ub.uni-rostock.de/api/speeches/"

    def __init__(self, work: Literal["Iliad", "Odyssey"] = "Iliad"):
        self.page = 1
        self.speeches = []

        if work == "Iliad":
            self.work = 1
        elif work == "Odyssey":
            self.work = 2

    def get_speeches(self):
        _url = f"{self.base_url}?work={self.work}&page={self.page}"

        while True:
            r = requests.get(_url)
            response = r.json()

            self.speeches += response["results"]

            _url = response["next"]

            if _url is None:
                break

            time.sleep(0.3)

        return self.speeches


def get_lines_from_dataframe(
    df: pl.DataFrame, title: str, lines: list[int]
):
    rows = (
        df.filter(
            pl.col("title").eq(title)
            & pl.col("book_n").is_in([l[0] for l in lines])
            & pl.col("n").ge(lines[0][1])
            & pl.col("n").le(lines[1][1])
        )
        .select(pl.col("text"))
    ).rows()

    return [r[0] for r in rows]


def get_speeches_by_title_and_speaker(
    df: pl.DataFrame, title: Literal["Iliad", "Odyssey"]
):
    with open(f"./dices_speeches/{title.lower()}_speeches.json") as f:
        speeches = json.load(f)

    speeches_by_speaker = {}

    for speech in speeches:
        speaker_id = "-".join([f"{s["name"]}@{s["id"]}" for s in speech["spkr"]])

        if speaker_id not in speeches_by_speaker:
            speeches_by_speaker[speaker_id] = []

        first_line = [int(n) for n in speech["l_fi"].split(".")]
        last_line = [int(n) for n in speech["l_la"].split(".")]

        lines = get_lines_from_dataframe(df, title, [first_line, last_line])

        speeches_by_speaker[speaker_id] += lines

    return speeches_by_speaker


def write_speeches_as_docs():
    homer_df = pl.read_parquet("./homer.parquet").cast(
        {"book_n": pl.UInt8, "n": pl.UInt16}
    )

    iliad = get_speeches_by_title_and_speaker(homer_df, "Iliad")

    for speaker, lines in iliad.items():
    	with open(f"./homeric_corpus/Iliad_{speaker}.txt", "w") as f:
            f.write("\n".join(lines))

    odyssey = get_speeches_by_title_and_speaker(homer_df, "Odyssey")

    for speaker, lines in odyssey.items():
        with open(f"./homeric_corpus/Odyssey_{speaker}.txt", "w") as f:
            f.write("\n".join(lines))

def speeches_to_conllu():
	pass

