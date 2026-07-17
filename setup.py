from setuptools import setup
import io
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def get_version():
    pkg_info = os.path.join(here, 'PKG-INFO')
    if os.path.isfile(pkg_info):
        with open(pkg_info) as f:
            for line in f:
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    version_file = os.path.join(here, 'version.py')
    if os.path.isfile(version_file):
        with open(version_file) as f:
            match = re.search(r'__version__\s*=\s*["\'](.+?)["\']', f.read())
            if match:
                return match.group(1)
    return '0.2.2'


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
