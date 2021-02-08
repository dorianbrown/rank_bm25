# Rank-BM25: A two line search engine

![Build Status](https://github.com/dorianbrown/rank_bm25/workflows/pytest/badge.svg)
[![PyPI version](https://badge.fury.io/py/rank-bm25.svg)](https://badge.fury.io/py/rank-bm25)
[![DOI](https://zenodo.org/badge/166720547.svg)](https://zenodo.org/badge/latestdoi/166720547)

A collection of algorithms for querying a set of documents and returning the ones most relevant to the query. The most common use case for these algorithms is, as you might have guessed, to create search engines.

So far the algorithms that have been implemented are:
- [x] Okapi BM25
- [x] BM25L
- [x] BM25+
- [ ] BM25-Adpt
- [ ] BM25T 

These algorithms were taken from [this paper](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf), which gives a nice overview of each method, and also benchmarks them against each other. A nice inclusion is that they compare different kinds of preprocessing like stemming vs no-stemming, stopword removal or not, etc. Great read if you're new to the topic. 

## Installation
The easiest way to install this package is through `pip`, using
```bash
pip install rank_bm25
```
If you want to be sure you're getting the newest version, you can install it directly from github with
```bash
pip install git+ssh://git@github.com/dorianbrown/rank_bm25.git
```

## Usage
For this example we'll be using the `BM25Okapi` algorithm, but the others are used in pretty much the same way.

### Initalizing

First thing to do is create an instance of the BM25 class, which reads in a corpus of text and does some indexing on it:
```python
from rank_bm25 import BM25Okapi

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
# <rank_bm25.BM25Okapi at 0x1047881d0>
```
Note that this package doesn't do any text preprocessing. If you want to do things like lowercasing, stopword removal, stemming, etc, you need to do it yourself. 

The only requirements is that the class receives a list of lists of strings, which are the document tokens.

### Ranking of documents

Now that we've created our document indexes, we can give it queries and see which documents are the most relevant:
```python
query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
# array([0.        , 0.93729472, 0.        ])
```
Good to note that we also need to tokenize our query, and apply the same preprocessing steps we did to the documents in order to have an apples-to-apples comparison

Instead of getting the document scores, you can also just retrieve the best documents with
```python
bm25.get_top_n(tokenized_query, corpus, n=1)
# ['It is quite windy in London']
```
And that's pretty much it!
