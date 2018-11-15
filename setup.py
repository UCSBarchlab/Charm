import os
from pathlib import Path
from shutil import copy, rmtree
from sys import platform

from setuptools import setup, find_packages
from setuptools.command.install import install


class RegisterKernel(install):
    def run(self):
        super().run()
        if platform == 'linux':
            kernel_directory = os.path.expanduser("~/.local/share/jupyter/kernels/Charm")
        elif platform == 'windows':
            kernel_directory = os.path.expandvars("%APPDATA%\jupyter\kernels\\Charm")
        elif platform == 'darwin':
            kernel_directory = os.path.expanduser("~/Library/Jupyter/kernels/Charm")
        else:
            raise RuntimeError("Operation system not identified")
        kernel_directory = Path(kernel_directory)
        if kernel_directory.exists():
            rmtree(kernel_directory)
        kernel_directory.mkdir(parents=True)
        original_kernel_file = Path(__file__).parent / "Charm" / "scripts" / "kernel.json"
        copy(str(original_kernel_file), kernel_directory)


setup(
    name='Charm',
    install_requires=[
        'mcerp3',
        'ipykernel',
        'numpy',
        'networkx',
        'sympy',
        'pyparsing',
        'matplotlib',
        'z3',
        'pint',
        'pandas'
    ],
    packages=find_packages(),
    cmdclass={
        'install': RegisterKernel
    }
)
