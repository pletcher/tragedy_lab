import os
import re
import string
import unicodedata

import polars as pl

from lxml import etree
from transliterate import translit

DIR = "data"
NAMESPACES = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

def replace_sigmas(s: str):
    return re.sub(r"ς|c", "σ", s)


def remove_combining_fluent(string: str) -> str:
    """
    Source: https://gist.github.com/luizomf/54b58615cd674db44153470c369a8284
    """
    normalized = unicodedata.normalize('NFD', string)
    return ''.join(
        [l for l in normalized if not unicodedata.combining(l)]
    ).casefold()


def to_urn(s: str):
    return f"urn:cts:greekLit:{s.replace('.xml', '')}"


FILES = [
    (os.path.join(DIR, f), to_urn(f))
    for f in os.listdir(DIR)
    if os.path.isfile(os.path.join(DIR, f)) and f.endswith(".xml")
]

speakers = set()

def get_dramatist(urn: str):
    if "tlg0006" in urn: return "Euripides"

    if "tlg0011" in urn: return "Sophocles"

    if "tlg0085" in urn: return "Aeschylus"



def clean_token(s: str):
    no_punct = s.translate(str.maketrans('', '', string.punctuation))
    no_grave_accute = remove_combining_fluent(no_punct)
    normal_sigmas = replace_sigmas(no_grave_accute)

    return normal_sigmas



def iter_lines(title, urn, tree):
    rows = []

    for l in tree.iterfind(".//tei:l", namespaces=NAMESPACES):
        if l.text is not None:
            n = l.xpath("./@n")
            speaker = l.xpath("../tei:speaker//text()", namespaces=NAMESPACES)

            if len(speaker) > 0:
                speaker = speaker[0].strip().replace(".", "")

                if speaker != "":
                    speakers.add(speaker)
                    text = l.text.strip()
                    tokens = [clean_token(t) for t in text.split()]

                    row = {
                        "n": n[0],
                        "urn": urn,
                        "dramatist": get_dramatist(urn),
                        "title": title,
                        "speaker": speaker,
                        "text": text,
                        "tokens": tokens
                    }

                    rows.append(row)

    return rows

data = []

for f, urn in FILES:
    tree = etree.parse(f)
    title = tree.xpath("//tei:titleStmt/tei:title/text()", namespaces=NAMESPACES)[0]
    lines = iter_lines(title, urn, tree)
    data += lines

df = pl.DataFrame(data)

df.write_parquet('./greek-tragedy-by-line.parquet')

for row in df.group_by("dramatist", "title").agg(pl.col("text")).iter_rows():
    title = f"{row[0]}_{row[1].replace(" ", "-")}"
    text = row[2]

    with open(f"./corpus/{title}.txt", "w+") as f:
        for line in text:
            f.write(f"{line}\n")


def lemmatize():
    import spacy

    nlp = spacy.load("grc_perseus_lg")

    txts = [f"./corpus/{f}" for f in os.listdir("./corpus") if f.endswith(".txt")]

    for txt in txts:
        with open(txt) as f:
            doc = nlp(f.read())
            lemmata = [token.lemma_ for token in doc]
            out = txt.replace(".txt", ".lemmatized.txt")

            with open(out, 'w+') as g:
                for lemma in lemmata:
                    g.write(f"{lemma}\n")