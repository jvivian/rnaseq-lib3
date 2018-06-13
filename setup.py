from setuptools import setup, find_packages

setup(name='rnaseq-lib3',
      version='1.0a1',
      description='Library of convenience functions related to current research',
      url='http://github.com/jvivian/rnaseq-lib3',
      author='John Vivian',
      author_email='jtvivian@gmail.com',
      license='MIT',
      package_dir={'': 'src'},
      packages=find_packages('src'), install_requires=['requests', 'scikit-learn', 'numba', 'annoy', 'numpy',
                                                       'synapseclient', 'pandas', 'holoviews'])
