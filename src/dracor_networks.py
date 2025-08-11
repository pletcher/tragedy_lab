import json
import time

from pathlib import Path

import pandas as pd
import requests

from lxml import etree
from tqdm import tqdm

NAMESPACES = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}


TRAGEDIANS = {
    "Aeschylus",
    "Euripides",
    "Sophocles",
}

network_json_file = Path("./tragedy_networks/tragedies.json")


def _find(tree, path):
    return tree.find(path, namespaces=NAMESPACES)


def _xpath(tree, path):
    return tree.xpath(path, namespaces=NAMESPACES)


class Character:
    def __init__(self, tree: etree._ElementTree, name: str):
        self.tree = tree
        self.name = name
        self.lines = self.find_lines()

    def find_lines(self):
        lines = []

        for lineset in self.tree.iterfind(
            f".//tei:sp[@who='#{self.name}']", namespaces=NAMESPACES
        ):
            for line in lineset.iterfind("./tei:l", namespaces=NAMESPACES):
                lines.append({"text": line.text, "n": line.get("n")})

        return lines


class DraCorNetwork:
    base_url = "https://dracor.org/api"

    def __init__(self, corpus_name: str):
        self.corpus_name = corpus_name

        self.corpus = None

    def get_corpus(self):
        r = requests.get(f"{self.base_url}/corpora/{self.corpus_name}")

        self.corpus = r.json()

        return self.corpus

    def get_cast(self, play_name: str):
        """
        Get the network data for the cast of the given play.
        Uses the name found at drama["name"], *not* drama["title"]
        """
        r = requests.get(
            f"{self.base_url}/corpora/{self.corpus_name}/play/{play_name}/cast"
        )

        return r.json()


def get_tragedy_networks():
    network_data = None

    if network_json_file.exists():
        with network_json_file.open() as f:
            network_data = json.load(f)
    else:
        network = DraCorNetwork("greek")

        # we only want tragedy for now
        dramas = [
            d
            for d in network.get_corpus()["dramas"]
            if d["authors"][0]["shortname"] in TRAGEDIANS
        ]

        for drama in tqdm(dramas):
            data = network.get_cast(drama["name"])

            drama["network_data"] = data
            time.sleep(0.1)

        with network_json_file.open("w") as f:
            json.dump(dramas, f, indent=2, ensure_ascii=False)

        network_data = dramas

    return network_data


def parse_file(f: str):
    tree = etree.parse(f)

    title = _find(tree, ".//tei:titleStmt/tei:title").text  # type: ignore
    dramatist = _find(tree, ".//tei:titleStmt/tei:author/tei:persName").text  # type: ignore
    personae = [
        Character(tree, c)
        for c in _xpath(
            tree, ".//tei:profileDesc/tei:particDesc/tei:listPerson/tei:person/@xml:id"  # type: ignore
        )
    ]


def prepare_dataframe():
    """
    Based on https://dracor-org.github.io/dracor-notebooks/catch-a-protagonist-in-dracor/catch-a-protagonist-in-dracor.html
    """
    with network_json_file.open() as f:
        tragedies = json.load(f)

    cols = [
        "id",
        "betweenness",
        "degree",
        "closeness",
        "weightedDegree",
        "eigenvector",
        "numOfScenes",
        "numOfSpeechActs",
        "numOfWords",
    ]

    # prepare the data for the data frame
    df_data = []

    for tragedy in tragedies:
        character_data = tragedy["network_data"]
        tragedy_id = tragedy["id"]
        tragedy_title = tragedy["title"]

        for character in character_data:
            c = dict(tragedy_id=tragedy_id, tragedy_title=tragedy_title)

            for key in cols:
                c[key] = character[key]

            df_data.append(c)

    # construct the data frame
    df = pd.DataFrame(df_data)

    return df


def rank_dataframe(df: pd.DataFrame):
    metrics_to_rank = [
        "degree",
        "closeness",
        "betweenness",
        "weightedDegree",
        "eigenvector",
        "numOfScenes",
        "numOfSpeechActs",
        "numOfWords",
    ]

    for metric in metrics_to_rank:
        df[f"{metric}_rank"] = df.groupby("tragedy_id")[metric].rank(
            method="min", ascending=False
        )

    ranks = [c for c in df.columns if c.endswith("rank")]

    df["centrality_rank_avg"] = df[ranks].sum(axis=1) / len(ranks)
    df["centrality_rank_std"] = df[ranks].std(axis=1)

    for metric in ["centrality_rank_avg", "centrality_rank_std"]:
        df[metric + "_rank"] = df.groupby("tragedy_id")[metric].rank(
            method="min", ascending=True
        )

    return df
