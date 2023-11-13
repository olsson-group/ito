from setuptools import find_packages, setup


setup(
    name='ito',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytorch_lightning',
        'torch_geometric',
        'mdshare',
        'mdtraj',
        'tqdm',
        'deeptime',
        'matplotlib'
    ],
)
