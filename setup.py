from setuptools import setup, find_packages

setup(
    name='pytorch-trainer',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/Hiroshiba/pytorch-trainer',
    author='Kazuyuki Hiroshiba',
    author_email='hihokaruta@gmail.com',
    description='PyTorch\'s Trainer like Chainer\'s Trainer',
    license='MIT License',
    install_requires=[
        'torch',
        'typing-extensions',
    ],
)
