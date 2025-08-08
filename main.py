import json

from src import write_homer_df, write_tragedy_df, write_corpus
from src.lemmatize import lemmatize
from src.homeric_speeches import DICESClient, write_speeches_as_docs

if __name__ == "__main__":
    write_speeches_as_docs()
