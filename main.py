import json

from src import write_homer_df, write_tragedy_df, write_corpus
from src.lda import plot
from src.normalization import to_conllu
from src.homeric_speeches import (
    DICESClient,
    write_speeches_to_conllu,
    write_speeches_to_docs,
)

if __name__ == "__main__":
    plot()
