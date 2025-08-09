import os
import re
import spacy
import unicodedata

from pathlib import Path
from tqdm import tqdm

from .util.stop_words import HARD_CODED_STOPS
from .util.transliterate import transliterate


def write_sentence_to_conllu(f, sent):
    f.write(f"# sentence: {re.sub(r"\s+", " ", sent.text.strip())}\n")

    sentence_tokens_with_index = list(enumerate([t for t in sent if not t.is_space]))

    for idx, t in sentence_tokens_with_index:
        id_ = idx + 1
        form = t.text.strip() or "_"
        lemma = t.lemma_ or "_"
        upos = t.pos_ or "_"
        xpos = t.tag_ or "_"
        feats = "_"

        if t.morph is not None and not t.is_punct:
            maybe_feats = str(t.morph)

            if maybe_feats.strip() != "":
                feats = maybe_feats

        head_id = 0
        deprel = "root"

        if t.head is not None:
            head_id = [h[0] for h in sentence_tokens_with_index if h[1].i == t.head.i]
            deprel = t.dep_

        # For some reason, CoNLL-U suggests "Yes" and "No" rather than True and False
        is_stop = "No"

        if t.is_stop or t.lemma in HARD_CODED_STOPS:
            is_stop = "Yes"

        is_punct = "No"

        if t.is_punct:
            is_punct = "Yes"

        misc = f"IsStop={is_stop}|IsPunct={is_punct}"

        f.write(
            unicodedata.normalize(
                "NFKC",
                f"{id_}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head_id}\t{deprel}\t_\t{misc}\n"
            )
        )

def to_conllu(directory: str = "./tragic_corpus", out_directory: str = "./tragic_conllu"):
    nlp = spacy.load("grc_proiel_trf")

    txts = [Path(f"{directory}/{f}") for f in os.listdir(directory) if f.endswith(".txt")]

    for txt in tqdm(txts):
        with txt.open() as f:
            doc = nlp(f.read())

            out = Path(f"./{out_directory}/{txt.name.replace('.txt', '.conllu')}")

            with out.open("w+") as g:
                for sent in doc.sents:
                    write_sentence_to_conllu(g, sent)
