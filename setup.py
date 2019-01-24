from setuptools import setup

setup(
   name='rank_bm25',
   version='1.0',
   description='Various BM25 algorithms for document ranking',
   author='D. Brown',
   author_email='dorianstuartbrown@gmail.com',
   license='LICENSE',
   py_modules=['rank_bm25'],
   install_requires=['numpy'],
)
