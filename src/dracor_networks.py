import json
import time

from pathlib import Path

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
    network_json_file = Path("./tragedy_networks/tragedies.json")

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


if __name__ == "__main__":
    get_tragedy_networks()
