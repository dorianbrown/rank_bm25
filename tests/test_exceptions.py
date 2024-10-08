import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pytest

from rank_bm25 import BM25
from exceptions import EmptyCorpusException


def test_empty_corpus():
    """Make sure that correct Exception is thrown when any algorithm initializes with an empty corpus"""
    with pytest.raises(EmptyCorpusException):
        BM25([])
