from rank_bm25.bm25 import BM25Atire, BM25L, BM25Plus

from sklearn.datasets import fetch_20newsgroups
from email.parser import Parser
from multiprocessing import Pool

import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

parser = Parser()
ls = LancasterStemmer()


def get_email_body(email_txt):
    msg = parser.parsestr(email_txt)
    return msg.get_payload()


def preprocess_doc(doc):
    doc = doc.strip()  # Remove leading/trailing ws
    doc = doc.lower()  # Lowercase all characters
    doc = re.sub(r'[^\w\s]', '', doc)  # remove punctuation
    doc = re.sub(r'\s+', ' ', doc)  # replace any whitespace characters with a space
    words = word_tokenize(doc, language='english')
    words = [w for w in words if w not in stopwords.words('english')]
    words = [ls.stem(w) for w in words]
    return words


news = fetch_20newsgroups()
email_body = [get_email_body(email) for email in news.data]
pool = Pool()
corpus = pool.map(preprocess_doc, email_body)


def test_corpus_loading():
    query_list = [
        "religious fanatics",
        "beefy computer specifications",
        "pain joint inflammation"
    ]

    bma = BM25Atire(corpus)
    bml = BM25L(corpus)
    bmp = BM25Plus(corpus)

    def evaluate_q(queries, model):
        print(f"\nEvaluation {model}", "\n", "-"*100)
        for q in queries:
            print("Query:", q)
            print(f"Normalized query: {preprocess_doc(q)}", "\n", "-"*100)
            for doc in model.get_top_n(preprocess_doc(q), email_body, 1):
                print(doc[:1000], "\n", "-"*80)

    evaluate_q(query_list, bma)
    evaluate_q(query_list, bml)
    evaluate_q(query_list, bmp)
