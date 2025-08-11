import json
import unicodedata

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyconll

from matplotlib.colors import LinearSegmentedColormap

from .normalization import to_conllu

MESSENGER_DIR = Path("./messenger_conllu")
MESSENGER_DOCS = [f for f in MESSENGER_DIR.iterdir() if f.suffix == ".conllu"]

TRAGEDY_DIR = Path("./tragic_conllu")
TRAGEDY_DOCS = [f for f in TRAGEDY_DIR.iterdir() if f.suffix == ".conllu"]


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


def plot_messenger_topics(data, title):
    names = list(data.keys())
    total_tokens = np.array([data[name]["total_tokens"] for name in names])
    topics = [k for k in data[names[0]].keys() if not k.startswith("total")]

    blues = mpl.colormaps["Blues"].resampled(10)
    reds = mpl.colormaps["Reds"].resampled(10)

    # colormaps are not particularly well-documented:
    # https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html#colormap-manipulation

    colors = list(blues(np.linspace(0.3, 1.0, 10))) + list(
        reds(np.linspace(0.3, 1.0, 10))
    )

    topic_counts = {}

    for topic in topics:
        counts = []

        for name in names:
            counts.append(data[name][topic] / data[name]["total_tokens"])

        topic_counts[topic] = np.array(counts)

    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(names))

    for topic, topic_count in topic_counts.items():
        topic_idx = topics.index(topic)

        p = ax.bar(
            names,
            topic_count,
            width=0.5,
            color=colors[topic_idx],
            label=topic,
            bottom=bottom,
        )
        bottom += topic_count

    ax.set_title(title)
    ax.legend(loc="upper right")

    ax.figure.autofmt_xdate()
    ax.autoscale()

    plt.show()


def messenger_topic_proportions_from_conllu():
    with open("./homeric-topics_NMF.json") as f:
        homeric_topics_nmf = json.load(f)

    with open("./tragic-topics_NMF.json") as f:
        tragic_topics_nmf = json.load(f)

    messengers = {}

    for f in MESSENGER_DOCS:
        play, character, _ = f.name.split("_")
        key = f"{play}_{character}"

        print(key)

        if key not in messengers:
            messengers[key] = {"total_tokens": 0}

            for topic_id in homeric_topics_nmf:
                messengers[key][f"Homer {topic_id}"] = 0

            for topic_id in tragic_topics_nmf:
                messengers[key][f"Tragedy {topic_id}"] = 0

        sentences = pyconll.load_from_file(f)  # type: ignore

        for sentence in sentences:
            for token in sentence:
                messengers[key]["total_tokens"] += 1

                for topic_id, features in homeric_topics_nmf.items():
                    for feature, weight in features:
                        if unicodedata.normalize(
                            "NFKC", feature
                        ) == unicodedata.normalize(
                            "NFKC", token.lemma  # type: ignore
                        ):
                            messengers[key][f"Homer {topic_id}"] += 1
                for topic_id, features in tragic_topics_nmf.items():
                    for feature, weight in features:
                        if unicodedata.normalize(
                            "NFKC", feature
                        ) == unicodedata.normalize(
                            "NFKC", token.lemma  # type: ignore
                        ):
                            messengers[key][f"Tragedy {topic_id}"] += 1

    plot_messenger_topics(messengers, "Proportion of topics by messenger")


# use tragedy as a control
def tragedy_topic_proportions_from_conllu():
    with open("./homeric-topics_NMF.json") as f:
        homeric_topics_nmf = json.load(f)

    with open("./tragic-topics_NMF.json") as f:
        tragic_topics_nmf = json.load(f)

    tragedies = {}

    for f in TRAGEDY_DOCS:
        dramatist, play, _ = f.name.split("_")
        key = f"{dramatist}_{play}"

        print(key)

        if key not in tragedies:
            tragedies[key] = {"total_tokens": 0}

            for topic_id in homeric_topics_nmf:
                tragedies[key][f"Homer {topic_id}"] = 0

            for topic_id in tragic_topics_nmf:
                tragedies[key][f"Tragedy {topic_id}"] = 0

        sentences = pyconll.load_from_file(f)  # type: ignore

        for sentence in sentences:
            for token in sentence:
                tragedies[key]["total_tokens"] += 1

                for topic_id, features in homeric_topics_nmf.items():
                    for feature, weight in features:
                        if unicodedata.normalize(
                            "NFKC", feature
                        ) == unicodedata.normalize(
                            "NFKC", token.lemma  # type: ignore
                        ):
                            tragedies[key][f"Homer {topic_id}"] += 1
                for topic_id, features in tragic_topics_nmf.items():
                    for feature, weight in features:
                        if unicodedata.normalize(
                            "NFKC", feature
                        ) == unicodedata.normalize(
                            "NFKC", token.lemma  # type: ignore
                        ):
                            tragedies[key][f"Tragedy {topic_id}"] += 1

    plot_messenger_topics(tragedies, "Proportion of topics by tragedy")


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
