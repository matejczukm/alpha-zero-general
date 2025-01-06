from setuptools import setup, find_packages

setup(
    name='polygonal_packit',
    version='0.2.1',
    author='Michał Matejczuk, Piotr Kosakowski',
    description='Hexagonal and triangular version of game Pack It! along with Alpha Zero General-based RL algorithms.',
    long_description=open('README.md').read(),
    url='https://github.com/matejczukm/alpha-zero-general',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    license='MIT',
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'huggingface_hub',
        'tensorflow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
