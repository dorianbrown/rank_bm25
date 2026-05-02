import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

from version import get_version

here = os.path.abspath(os.path.dirname(__file__))

short_description = 'Various BM25 algorithms for document ranking'

try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = short_description


class build_py(_build_py):
    def run(self):
        _build_py.run(self)
        self._compile_csc_accel()

    def _compile_csc_accel(self):
        src = os.path.join(here, "_csc_accum.c")
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        name = "_csc_accum" + ext
        cc = "clang" if sys.platform == "darwin" else "gcc"
        cc_flags = [
            cc,
            "-Ofast",
            "-march=native",
            "-ffast-math",
            "-shared",
            "-fPIC",
            "-o",
        ]

        for out_dir in (self.build_lib, here):
            out = os.path.join(out_dir, name)
            try:
                subprocess.check_call(
                    cc_flags + [out, src],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass


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
    install_requires=['numpy', 'scipy'],
    extras_require={
        'dev': [
            'pytest'
        ]
    },
    cmdclass={"build_py": build_py},
)
