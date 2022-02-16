from setuptools import setup
import io
import os

from version import get_version

here = os.path.abspath(os.path.dirname(__file__))

short_description = 'Various BM25 algorithms for document ranking'

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = short_description

setup(
    name='rank_bm25',
    version=get_version(),
    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='D. Brown',
    author_email='dorianstuartbrown@gmail.com',
    url="https://github.com/dorianbrown/rank_bm25",
    license='Apache2.0',
    py_modules=['rank_bm25'],
    install_requires=['numpy'],
    extras_require={
        'dev': [
            'pytest'
        ]
    }
)
