from setuptools import setup, find_packages

setup(
    name='ECG_processing',
    version='0.1',
    packages=find_packages(),
    description='A library for ECG signal processing',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ecg_lib',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
    ],
)
