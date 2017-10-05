from setuptools import setup
from distutils.extension import Extension

ext_modules = [ ]
ext_modules += [
#        Extension("boxsimu.solver", [ "boxsimu/solver.pyx" ],
#            include_dirs=['.']),
#		Extension("boxsimu.system", [ "boxsimu/system.pyx" ],
#            include_dirs=['.']),
]

setup(
      name='boxsimu',
      version='0.1',
      description='Simple box modelling framework.',
      url='',
      author='Mathias Aschwanden',
      author_email='mathias.aschwanden@gmail.com',
      license='MIT',
      packages=['boxsimu'],
      zip_safe=False,
      include_package_data=True,
      ext_modules=ext_modules,
) 
