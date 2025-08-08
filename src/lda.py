import pyconll
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CONLLU_DIR = Path("./conllu")

CONLLU_FILES = [f for f in CONLLU_DIR.iterdir() if f.suffix == ".conllu"]


def load_corpus(fs: list[Path] = CONLLU_FILES) -> list[str]:
    corpus = []

    for f in CONLLU_FILES:
        print(f)
        tokens = pyconll.load_from_file(f)  # type: ignore
        print(tokens)

    return corpus


load_corpus()
