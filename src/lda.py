import unicodedata

from pathlib import Path
from time import time
from typing import Dict, Literal

import pyconll
import matplotlib.pyplot as plt

from pyconll.unit.token import Token
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .util.stop_words import HARD_CODED_STOPS

# source and inspiration:
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

init = "nndsvda"
n_components = 10
n_features = 1000
n_top_words = 20

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features)

# filter stop words based on the IsStop MISC value in the CoNLL-U files
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=n_features,
)


def _should_include_token(token: Token) -> bool:
    return (
        type(token.misc["IsStop"]) == set
        and "No" in token.misc["IsStop"]
        and type(token.misc["IsPunct"]) == set
        and "No" in token.misc["IsPunct"]
        and (unicodedata.normalize("NFKC", str(token.lemma)) not in HARD_CODED_STOPS)
    )


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=1)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": "x-small"})
        ax.tick_params(axis="both", which="major", labelsize=10)

        for i in "top right left".split():
            ax.spines[i].set_visible(False)

        ax.figure.autofmt_xdate()
        fig.suptitle(title, fontsize=20)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def load_corpus(fs: list[Path]) -> Dict[str, list[Token]]:
    corpus = {}

    for f in fs:
        sentences = pyconll.load_from_file(f)  # type: ignore

        corpus_key = f.name.replace(f.suffix, "")
        corpus[corpus_key] = []

        for sentence in sentences:
            for token in sentence:
                if _should_include_token(token):
                    corpus[corpus_key].append(token)

    return corpus


def run_nmf(tfidf, beta_loss="frobenius"):
    return NMF(
        n_components=n_components,  # type: ignore
        random_state=1,
        init=init,
        beta_loss=beta_loss,
        alpha_W=0.00005,
        alpha_H="same",
        l1_ratio=1,
    ).fit(tfidf)


def vectorize_with_tf(samples):
    return tf_vectorizer.fit_transform(samples)


def vectorize_with_tfidf(samples):
    return tfidf_vectorizer.fit_transform(samples)


def plot(directory: Literal["homeric_conllu", "messenger_conllu", "tragic_conllu"]):
    CONLLU_DIR = Path(directory)
    CONLLU_FILES = [f for f in CONLLU_DIR.iterdir() if f.suffix == ".conllu"]

    t0 = time()
    corpus = load_corpus(CONLLU_FILES)
    print("Corpus loaded in %0.3fs" % (time() - t0))

    samples = []

    for speaker, tokens in corpus.items():
        samples.append(" ".join([str(t.lemma) for t in tokens]))

    t1 = time()
    tfidf = vectorize_with_tfidf(samples)
    print("TF-IDF vectorizer run in %0.3fs" % (time() - t1))

    t2 = time()
    tf = vectorize_with_tf(samples)
    print("TF vectorizer run in %0.3fs" % (time() - t2))

    t3 = time()
    nmf_frobenius = run_nmf(tfidf)
    print("NMF run in %0.3fs" % (time() - t3))

    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    plot_top_words(
        nmf_frobenius,
        tfidf_feature_names,
        n_top_words,
        "Topics in NMF model (Frobenius norm)",
    )

    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=50,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    tf_feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")
