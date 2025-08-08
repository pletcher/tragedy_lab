import time

from typing import Literal

import requests


class DICESClient:
    base_url = "http://dices.ub.uni-rostock.de/api/speeches/"

    def __init__(self, work: Literal["Iliad", "Odyssey"] = "Iliad"):
        self.page = 1
        self.speeches = []

        if work == "Iliad":
            self.work = 1
        elif work == "Odyssey":
            self.work = 2

    def get_speeches(self):
        _url = f"{self.base_url}?work={self.work}&page={self.page}"

        while True:
            r = requests.get(_url)
            response = r.json()

            self.speeches += response["results"]

            _url = response["next"]

            if _url is None:
                break

            time.sleep(0.3)

        return self.speeches
