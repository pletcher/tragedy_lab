import json

from src import write_homer_df, write_tragedy_df, write_corpus
from src.lemmatize import lemmatize
from src.homeric_speeches import (
    DICESClient,
    write_speeches_to_conllu,
    write_speeches_to_docs,
)

if __name__ == "__main__":
    write_speeches_to_conllu()
