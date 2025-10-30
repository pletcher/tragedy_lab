import json

from src import write_homer_df, write_tragedy_df, write_corpus
from src.normalization import to_conllu
from src.dracor_networks import prepare_dataframe, rank_dataframe
from src.homeric_speeches import (
    DICESClient,
    write_speeches_to_conllu,
    write_speeches_to_docs,
)
from src.messenger_speeches import (
    messenger_topic_proportions_from_conllu,
    tragedy_topic_proportions_from_conllu,
    write_messenger_speeches_to_conllu,
    write_messenger_speeches_to_docs,
)
from src.topic_modeling import plot

if __name__ == "__main__":
    df = prepare_dataframe()
    df = rank_dataframe(df)

    df.to_csv("./ranked_dracor_networks_in_tragedy.csv")
