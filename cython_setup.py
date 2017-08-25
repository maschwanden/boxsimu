from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [Extension("*", ["*.pyx"])]

setup(
	name = 'BoxSimu Package',
    ext_modules = cythonize(extensions)
)
