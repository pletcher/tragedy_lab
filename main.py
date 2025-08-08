import json

from src import write_homer_df, write_tragedy_df, write_corpus
from src.lemmatize import lemmatize
from src.homeric_speeches import DICESClient

if __name__ == "__main__":
    iliad_client = DICESClient("Iliad")

    iliad_speeches = iliad_client.get_speeches()

    with open("homeric_speeches/iliad_speeches.json", "w") as f:
        f.write(json.dumps(iliad_speeches, indent=4))

    odyssey_client = DICESClient("Odyssey")

    odyssey_speeches = odyssey_client.get_speeches()

    with open("homeric_speeches/odyssey_speeches.json", "w") as f:
        f.write(json.dumps(odyssey_speeches, indent=4))
