from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="src.cython.frame_enhancer_cython",
        sources=["src/cython/frame_enhancer_cython.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
    Extension(
        name="src.cython.change_detector_cython",
        sources=["src/cython/change_detector_cython.pyx"],
        include_dirs=[np.get_include()],
        language="c++"
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
