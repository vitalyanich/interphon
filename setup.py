from setuptools import setup


setup_requires = []
install_requires = ['Click', 'numpy', 'matplotlib']  # include 'ase' to use the visualization of vibration mode.

packages_interphon = ['InterPhon',
                      'InterPhon.core',
                      'InterPhon.error',
                      'InterPhon.inout',
                      'InterPhon.util',
                      'InterPhon.analysis', ]

scripts_interphon = ['scripts/interphon.py', ]


if __name__ == '__main__':
    setup(name='InterPhon',
          version='0.1',
          description='This is the InterPhon package.',
          author='In Won Yeu',
          author_email='yiw0121@snu.ac.kr',
          packages=packages_interphon,
          install_requires=install_requires,  # The package written here will be installed with the current package.
          python_requires='>=3',
          setup_requires=setup_requires,
          # scripts=scripts_interphon,
          entry_points={'console_scripts': ['interphon = InterPhon.interphon:main', ], },
          )