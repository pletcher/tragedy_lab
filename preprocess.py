import csv
import os
import re
import string
import unicodedata

import polars as pl

from lxml import etree

HOMER_DIR = "homer_xml"
TRAGEDY_DIR = "tragedy_xml"

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


TRAGEDY_FILES = [
    (os.path.join(TRAGEDY_DIR, f), to_urn(f))
    for f in os.listdir(TRAGEDY_DIR)
    if os.path.isfile(os.path.join(TRAGEDY_DIR, f)) and f.endswith(".xml")
]

HOMER_FILES = [
    (os.path.join(HOMER_DIR, f), to_urn(f))
    for f in os.listdir(HOMER_DIR)
    if os.path.isfile(os.path.join(HOMER_DIR, f)) and f.endswith(".xml")
]

PERSONAE = []

with open("speakers_with_genders.csv", newline="") as f:
    csv_reader = csv.DictReader(f, delimiter=",")

    for row in csv_reader:
        PERSONAE.append(row)

speakers = set()

def get_dramatist(urn: str):
    if "tlg0006" in urn: return "Euripides"

    if "tlg0011" in urn: return "Sophocles"

    if "tlg0085" in urn: return "Aeschylus"


def get_gender(dramatist: str, title: str, speaker: str):
    by_dramatist = [r for r in PERSONAE if r['dramatist'] == dramatist]
    by_title = [r for r in by_dramatist if r['title'] == title]
    char = [r for r in by_title if r['speaker'] == speaker][0]

    return char['gender']


def clean_token(s: str):
    no_punct = s.translate(str.maketrans('', '', string.punctuation))
    no_grave_accute = remove_combining_fluent(no_punct)
    normal_sigmas = replace_sigmas(no_grave_accute)

    return normal_sigmas


TITLES = {
    "Ichneutae": "Ichneutae",
    "Ἀγαμέμνων": "Agamemnon",
    "Αἴας": "Ajax",
    "Ἄλκηστις": "Alcestis",
    "Ἀνδρομάχη": "Andromache",
    "Ἀντιγόνη": "Antigone",
    "Βάκχαι": "Bacchae",
    "Ἑκάβη": "Hecuba",
    "Ἑλένη": "Helen",
    "Ἑπτὰ ἐπὶ Θήβας": "Seven Against Thebes",
    "Εὐμενίδες": "Eumenides",
    "Ἠλέκτρα": "Electra",
    "Ἡρακλεῖδαι": "Heracleidae",
    "Ἡρακλῆς": "Heracles",
    "Ἱκέτιδες": "Suppliants",
    "Ἱππόλυτος": "Hippolytus",
    "Ἰφιγένεια ἐν Αὐλίδι": "IA",
    "Ἰφιγένεια ἐν Ταύροις": "IT",
    "Ἴων": "Ion",
    "Κύκλωψ": "Cyclops",
    "Μήδεια": "Medea",
    "Οἰδίπους ἐπὶ Κολωνῷ": "OC",
    "Οἰδίπους Τύραννος": "OT",
    "Ὀρέστης": "Orestes",
    "Πέρσαι": "Persians",
    "Προμηθεὺς δεσμώτης": "Prometheus Bound",
    "Ῥῆσος": "Rhesus",
    "Τραχίνιαι": "Trachiniae",
    "Τρῳάδες": "Trojan Women",
    "Φιλοκτήτης": "Philoctetes",
    "Φοίνισσαι": "Phoenician Women",
    "Χοηφóρoι": "Choephoroi",
}


def iter_homer_lines(urn, tree):
    rows = []

    book_n = "1"

    title = ""

    if "tlg0012.tlg001" in urn:
        title = "Iliad"
    elif "tlg0012.tlg002" in urn:
        title = "Odyssey"

    if title == "":
        raise RuntimeError(f"Invalid URN: {urn}")

    for l in tree.iterfind(".//tei:l", namespaces=NAMESPACES):
        text = ''.join(l.itertext()).strip()

        if text is not None:
            n = l.xpath("./@n")[0]
            book_el = l.getparent()

            possible_book_n = book_el.attrib.get('n')
            if possible_book_n is not None and possible_book_n != book_n:
                book_n = possible_book_n

            row = dict(
                urn=f"{urn}:{book_n}.{n}",
                title=title,
                book_n=book_n,
                n=n,
                text=text
            )

            rows.append(row)

    return rows

def iter_tragedy_lines(title, urn, tree):
    rows = []

    for l in tree.iterfind(".//tei:l", namespaces=NAMESPACES):
        text = ''.join(l.itertext()).strip()

        if text is not None:
            n = l.xpath("./@n")
            speaker = l.xpath("../tei:speaker//text()", namespaces=NAMESPACES)

            if len(speaker) > 0:
                speaker = speaker[0].strip().replace(".", "")

                if speaker != "":
                    speakers.add(speaker)
                    tokens = [clean_token(t) for t in text.split()]
                    dramatist = get_dramatist(urn)

                    row = {
                        "n": n[0],
                        "urn": urn,
                        "dramatist": dramatist,
                        "title": TITLES[title],
                        "speaker": speaker,
                        "gender": get_gender(dramatist, title, speaker),
                        "text": text,
                        "tokens": tokens
                    }

                    rows.append(row)

    return rows


def create_tragedy_df():
    data = []

    for f, urn in TRAGEDY_FILES:
        tree = etree.parse(f)
        title = tree.xpath("//tei:titleStmt/tei:title/text()", namespaces=NAMESPACES)[0]
        lines = iter_tragedy_lines(title, urn, tree)
        data += lines

    df = pl.DataFrame(data)

    return df

def create_homer_df():
    data = []

    for f, urn in HOMER_FILES:
        tree = etree.parse(f)
        lines = iter_homer_lines(urn, tree)
        data += lines
    
    df = pl.DataFrame(data)

    return df


def write_homer_df():
    df = create_homer_df()

    df.write_parquet('./homer.parquet')


def write_tragedy_df():
    df = create_tragedy_df()

    df.write_parquet('./greek-tragedy-by-line_with-gender.parquet')

def write_corpus():
    df = create_tragedy_df()

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

if __name__ == "__main__":
    write_tragedy_df()