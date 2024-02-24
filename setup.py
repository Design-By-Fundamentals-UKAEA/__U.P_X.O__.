from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 3 - Alpha',  # If package is still in early development
    #'Development Status :: 4 - Beta',  # If package that's getting closer to a stable release
    #'Development Status :: 5 - Production/Stable',  # If stable package

    # Specify the audience and topic of your package
    'Intended Audience :: Computational Materials Science/Research',  # Since it's relevant to computational materials science
    'Intended Audience :: Crystal plasticity based finite element method',
    'Intended Audience :: Grain growth kinetics',
    'Intended Audience :: Materials for Fusion Science/Research',  # Since it's relevant to scientific research
    'Intended Audience :: Developers',  # If it's also meant for developers working on FE simulation or material science projects
    'Topic :: Scientific/Engineering',  # General category for scientific/engineering tools
    'Topic :: Scientific/Engineering :: Physics',  # If it involves physical principles, especially in material science
    'Topic :: Scientific/Engineering :: Visualization',  # If the package includes visualization of grain structures

    # Specify the license
    'License :: OSI Approved :: MIT License',  # If you're using the MIT License
    # 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  # If you're using the GPLv3

    # Specify supported Python versions
    'Programming Language :: Python :: 3',  # If compatible with Python 3
    'Programming Language :: Python :: 3.6',  # And other specific versions it's been tested on
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

setup(
    name='upxo',
    version='1.26.1',
    author='Dr. Sunil Anandatheertha',
    author_email='sunilanandatheertha@gmail.com',
    description='A package for grain structure generation, analysis, and export to FE simulation software',
    long_description=open('readme.md').read(),
    url='https://github.com/SunilAnandatheertha/upxo_private/',
    download_url='https://github.com/SunilAnandatheertha/upxo_private/',
    project_urls= {'Source Code': 'https://github.com/SunilAnandatheertha/upxo_private/tree/upxo.v.1.26.1/src',
                   'Documentation': 'https://github.com/SunilAnandatheertha/upxo_private/tree/upxo.v.1.26.1/docs',
                   },
    install_requires='', # A list of dependencies required to install and run your package. These will be installed by pip when your package is installed
    python_requires='>=3.6',
    classifiers=classifiers,
    package_dir={'': 'src'},  # This tells setuptools that packages are under 'src'
    packages=find_packages(where='src'),  # This will find the package inside the 'src' directory
)
