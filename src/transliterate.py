import unicodedata

greek_to_latin = {
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "e",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "ph",
    "χ": "ch",
    "ψ": "ps",
    "ω": "o",
}

greek_to_latin_caps = dict([(k.upper(), v.upper()) for k, v in greek_to_latin.items()])

transliteration_map = {**greek_to_latin, **greek_to_latin_caps}


def remove_combining(s: str) -> str:
    """
    Source: https://gist.github.com/luizomf/54b58615cd674db44153470c369a8284
    """
    normalized = unicodedata.normalize("NFD", s)

    return "".join([l for l in normalized if not unicodedata.combining(l)])


def transliterate(s: str) -> str:
    s = remove_combining(s)

    return "".join(
        [transliteration_map[l] if l in transliteration_map else l for l in s]
    ).title()
